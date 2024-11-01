//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "../Partitioning/GraphPartitioner.h"
#include "../Partitioning/SPNGraph.h"
#include "LoSPN/LoSPNOps.h"
#include "LoSPN/LoSPNPasses.h"
#include "LoSPNPassDetails.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Use.h"
#include "llvm/Support/raw_ostream.h"
#include <boost/graph/properties.hpp>
#include <boost/graph/subgraph.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/iterator_range_core.hpp>
#include <cassert>
#include <llvm/ADT/IndexedMap.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <optional>

namespace mlir {
namespace spn {
namespace low {

#define GEN_PASS_DEF_LOSPNTASKPARTIONING
#include "LoSPN/LoSPNPasses.h.inc"

using namespace mlir::spn::low::partitioning;

///
/// Pattern matching a LoSPN task and splitting the task
/// into multiple tasks by graph partitioning.
class PartitionTask : public OpRewritePattern<low::SPNTask> {
public:
  PartitionTask(MLIRContext *ctx, const TargetExecutionModel &targetModel,
                int maxTaskSize, bool schedule, bool decomposeTaskIntputs)
      : OpRewritePattern<low::SPNTask>(ctx, 1), targetModel_(targetModel),
        maxTaskSize_(maxTaskSize), schedule_(schedule),
        decomposeTaskInputs_(decomposeTaskIntputs) {}

  LogicalResult matchAndRewrite(SPNTask task,
                                PatternRewriter &rewriter) const override {
    SPNKernel kernel = task->getParentOfType<SPNKernel>();
    assert(kernel);
    {
      // Check if the kernel is already partitioned.
      size_t numTasksInKernel = 0;
      kernel.walk([&](SPNTask task) { ++numTasksInKernel; });
      if (numTasksInKernel > 1) {
        return rewriter.notifyMatchFailure(task,
                                           "Kernel is already partitioned");
      }
    }

    SPNBody body;
    {
      // Find the body of the task
      bool found = false;
      task.walk([&](SPNBody b) {
        body = b;
        assert(!found && "Task has more than one body");
        found = true;
      });
      assert(body);
    }

    {
      // Make sure that the number of nodes in the task is larger than the
      // maximum task size.
      unsigned numNodes = 0;
      body.walk([&](Operation *op) {
        if (!op->hasTrait<OpTrait::ConstantLike>()) {
          ++numNodes;
        }
      });
      if ((int)numNodes <= maxTaskSize_) {
        return rewriter.notifyMatchFailure(
            task, "Task is already smaller than the maximum task size");
      }
    }

    GraphPartitioner partitioning = partition(body, rewriter);

    // Connections between partitions
    InputMap connections;

    // Map arguments of the body's entry block to external inputs of the task.
    for (BlockArgument bodyBlockArg : body.getBody().getArguments()) {
      // Detect the BatchExtract producing this body block arg:
      SPNBatchExtract batchExtract = cast<SPNBatchExtract>(
          body->getOperand(bodyBlockArg.getArgNumber()).getDefiningOp());
      // Get the task block argument producing the input tensor of the
      // BatchExtract
      BlockArgument taskBlockArg =
          batchExtract.getInput().cast<BlockArgument>();

      // Get the kernel block argument producing the input tensor of the
      // task block argument.
      // Minus one because the first operand is the batch index.
      BlockArgument kernelBockArg =
          kernel.getBody().getArgument(taskBlockArg.getArgNumber() - 1);
      assert(kernelBockArg.isa<BlockArgument>());

      connections[bodyBlockArg] =
          InputInfo{kernelBockArg, std::nullopt, batchExtract.getStaticIndex()};
    }

    // Create a new LoSPN task for each cluster.
    for (auto &cluster : partitioning.clusters()) {
      // Skip clusters that only contain constants.
      auto vertices = boost::vertices(cluster);
      if (std::all_of(vertices.first, vertices.second, [&cluster](auto vertex) {
            return boost::get(SPNVertex_IsConstant(), cluster, vertex);
          }))
        continue;

      llvm::outs() << "Creating task for cluster\n";

      // Create a new task for this cluster.
      createTaskForPartition(cluster, rewriter, task.getLoc(),
                             task.getBatchSize(), connections, partitioning);
    }

    // Identify the task(s) producing the final result(s) of the original
    // task and replace the original task by the newly created tasks.
    SmallVector<Value> newResults;
    body.walk([&](SPNYield yield) {
      for (auto resVal : yield->getOperands()) {
        newResults.push_back(connections.lookup(resVal).tensor);
      }
    });
    rewriter.replaceOp(task, newResults);
    return mlir::success();
  }

private:
  struct InputInfo {
    Value tensor;
    std::optional<unsigned> rowIndex;
    std::optional<unsigned> colIndex;

    bool transposed() const {
      assert(rowIndex.has_value() ^ colIndex.has_value());
      return rowIndex.has_value();
    }
  };

  using InputMap = llvm::DenseMap<mlir::Value, InputInfo>;

  GraphPartitioner partition(SPNBody body, PatternRewriter &rewriter) const {
    GraphPartitioner partitioner(body, targetModel_, maxTaskSize_);
    partitioner.clusterGraph();
    partitioner.postprocessConstants(rewriter);
    // partitioner.scheduleGraphForBSP();
    return partitioner;
  }

  void createTaskForPartition(SPNGraph &partition, PatternRewriter &rewriter,
                              Location loc, unsigned batchSize,
                              InputMap &inputs,
                              GraphPartitioner &partitioner) const {
    // Step 1: Check if task for this partition has already been created.
    if (isTaskAlreadyCreated(partition, inputs, partitioner))
      return;

    // Inputs to the partition, either from outside or from other partitions.
    InputMap nonPartitionInputs;
    // Values that will become arguments of the new task.
    llvm::MapVector<Value, unsigned> inputArgs;
    unsigned inputArgIndex = 1; // First argument is the batch index.

    // Step 2: Collect all input values coming from outside arguments of the
    // original task.
    if (!decomposeTaskInputs_) {
      collectExternalInputs(partition, inputs, nonPartitionInputs, inputArgs,
                            inputArgIndex);
    } else {
      collectExternalInputsWithScheduling(partition, inputs, nonPartitionInputs,
                                          inputArgs, inputArgIndex, rewriter,
                                          loc);
    }

    // Step 3: Collect all input values coming from other partitions.
    collectInputsFromOtherPartitions(partition, rewriter, loc, batchSize,
                                     inputs, nonPartitionInputs, inputArgs,
                                     inputArgIndex, partitioner);

    // Step 4: Collect outputs this task will provide to other partitions.
    SmallVector<Value> nonPartitionOutputs;
    std::optional<Type> resultType;
    SmallVector<Type> bodyResults;
    collectTaskOutputs(partition, inputs, partitioner, resultType, bodyResults,
                       nonPartitionOutputs);

    // Step 5: Create the actual LoSPN task.
    auto outputType = createOutputType(resultType.value(), bodyResults.size());
    SmallVector<Value> taskInputs = getTaskInputs(inputArgs);
    auto task =
        createLoSPNTask(rewriter, loc, outputType, taskInputs, batchSize);

    {
      // Set the insertion point to the beginning of the task body.
      IRRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&task.getBody().front());

      // Step 6: Create a batch extract for each tensor argument of the new
      // task.
      llvm::DenseMap<Value, unsigned> inputIndices;
      SmallVector<Value> bodyInputs;
      llvm::IndexedMap<bool> hasLogType;
      createBatchExtractsForTaskInputs(task, nonPartitionInputs, inputArgs,
                                       inputIndices, bodyInputs, hasLogType,
                                       rewriter, loc);
      SPNBody body;
      {
        IRRewriter::InsertionGuard guard(rewriter);

        // Step 7: Copy operations into the task body. The insertion point is
        // set to the end of the body block.
        IRMapping mapper;
        body = createTaskBody(rewriter, loc, bodyResults, bodyInputs,
                              hasLogType, inputIndices, partition, mapper);

        // Step 8: Create SPNYield in the body and update inputs.
        updateInputsAndCreateYield(rewriter, loc, body, nonPartitionOutputs,
                                   inputs, task, mapper);
      }

      // Step 9: Create SPNBatchCollect and SPNReturn in the task.
      createBatchCollectAndReturn(rewriter, loc, task, body);
    }
  }

private:
  bool isTaskAlreadyCreated(SPNGraph &partition, InputMap &inputs,
                            GraphPartitioner &partitioner) const {
    auto edges_out = partitioner.edges_out(partition);
    return std::all_of(edges_out.begin(), edges_out.end(), [&](auto edge) {
      return inputs.contains(
          boost::get(SPNEdge_Value(), partitioner.graph(), edge));
    });
  }

  void collectExternalInputs(SPNGraph &partition, InputMap &inputs,
                             InputMap &nonPartitionInputs,
                             llvm::MapVector<Value, unsigned> &inputArgs,
                             unsigned &inputArgIndex) const {
    for (auto vertex : boost::make_iterator_range(boost::vertices(partition))) {
      if (boost::get(SPNVertex_UsesInput(), partition, vertex)) {
        auto *op = boost::get(SPNVertex_Operation(), partition, vertex);
        for (auto operand : op->getOperands()) {
          assert(inputs.count(operand) &&
                 "External input expected in input map");
          auto inputInfo = inputs[operand];
          nonPartitionInputs[operand] = inputInfo;
          if (!inputArgs.count(inputInfo.tensor)) {
            inputArgs.insert({inputInfo.tensor, inputArgIndex++});
          }
        }
      }
    }
  }

  void collectExternalInputsWithScheduling(
      SPNGraph &partition, InputMap &inputs, InputMap &nonPartitionInputs,
      llvm::MapVector<Value, unsigned> &inputArgs, unsigned &inputArgIndex,
      PatternRewriter &rewriter, Location loc) const {
    llvm::MapVector<Value,
                    std::set<std::pair<unsigned, ::mlir::detail::ValueImpl *>>>
        externalTensorUses;

    // Find all external tensors and the rows this task uses.
    for (auto vertex : boost::make_iterator_range(boost::vertices(partition))) {
      if (boost::get(SPNVertex_UsesInput(), partition, vertex)) {
        auto *op = boost::get(SPNVertex_Operation(), partition, vertex);
        for (Value operand : op->getOperands()) {
          assert(inputs.count(operand) &&
                 "External input expected in input map");
          auto inputInfo = inputs[operand];
          externalTensorUses[inputInfo.tensor].insert(
              {inputInfo.colIndex.value(), operand.getImpl()});
        }
      }
    }

    // Process each external tensor.
    for (auto &pair : externalTensorUses) {
      auto externalTensor = pair.first;
      auto &rows = pair.second;
      auto newTensor =
          createGatheredTensor(rewriter, loc, externalTensor, rows);

      unsigned internalRowIndex = 0;
      for (auto &row : rows) {
        Value internalTensor = row.second;
        nonPartitionInputs[internalTensor] =
            InputInfo{newTensor, std::nullopt, internalRowIndex++};
        if (!inputArgs.count(newTensor))
          inputArgs.insert({newTensor, inputArgIndex++});
      }
    }
  }

  /**
   * @brief Creates a gathered tensor from the given external tensor and
   * specified rows.
   *
   * This function generates a new tensor by gathering specific rows from an
   * external tensor. The rows to be gathered are specified by a set of pairs,
   * where each pair consists of an unsigned integer representing the row index
   * and the `Value` object representing the row in the SPNBody.
   *
   * @param rewriter The pattern rewriter used to create new operations.
   * @param loc The location information for the new operations.
   * @param externalTensor The external tensor from which rows will be gathered.
   * @param rows A set of pairs specifying the rows to be gathered. Each pair
   * contains an unsigned integer representing the row index and a pointer to a
   * ValueImpl object.
   * @return A new tensor containing the gathered rows from the external tensor.
   */
  Value createGatheredTensor(
      PatternRewriter &rewriter, Location loc, Value externalTensor,
      const std::set<std::pair<unsigned, ::mlir::detail::ValueImpl *>> &rows)
      const {
    auto tensorType = externalTensor.getType().cast<RankedTensorType>();
    auto newTensorType = RankedTensorType::get(
        {ShapedType::kDynamic, static_cast<long>(rows.size())},
        tensorType.getElementType());
    std::vector<int> externalIndices;
    for (auto &row : rows) {
      externalIndices.push_back(row.first);
    }
    auto indicesType = RankedTensorType::get(
        {static_cast<int64_t>(externalIndices.size())}, rewriter.getI32Type());
    auto indicesAttr = DenseIntElementsAttr::get(indicesType, externalIndices);
    auto indicesConst =
        rewriter.create<arith::ConstantOp>(loc, indicesType, indicesAttr);

    return rewriter.create<tensor::GatherOp>(
        loc, newTensorType, externalTensor, indicesConst, ArrayRef<int64_t>(1));
  }

  /**
   * @brief Collects inputs from other partitions and updates the input maps.
   *
   * This function iterates over the incoming edges of the given partition and
   * collects the input values from other partitions. It ensures that the inputs
   * are present in the input map after conversion and updates the non-partition
   * inputs and input arguments accordingly.
   *
   * @param partition The SPNGraph partition to collect inputs for.
   * @param rewriter The PatternRewriter used for creating tasks.
   * @param loc The location for the rewriter.
   * @param batchSize The batch size for task creation.
   * @param inputs The map of inputs for the current partition.
   * @param nonPartitionInputs The map of inputs that are not part of the
   * current partition.
   * @param inputArgs The map of input arguments with their corresponding
   * indices.
   * @param inputArgIndex The current index for input arguments.
   * @param partitioner The GraphPartitioner used to manage partitions and
   * edges.
   */
  void collectInputsFromOtherPartitions(
      SPNGraph &partition, PatternRewriter &rewriter, Location loc,
      unsigned batchSize, InputMap &inputs, InputMap &nonPartitionInputs,
      llvm::MapVector<Value, unsigned> &inputArgs, unsigned &inputArgIndex,
      GraphPartitioner &partitioner) const {
    for (auto globalInEdge : partitioner.edges_in(partition)) {
      Value value =
          boost::get(SPNEdge_Value(), partitioner.graph(), globalInEdge);
      if (!inputs.count(value)) {
        auto globalVertexFrom =
            boost::source(globalInEdge, partitioner.graph());
        auto otherPartition =
            find_cluster(globalVertexFrom, partitioner.graph());
        createTaskForPartition(otherPartition, rewriter, loc, batchSize, inputs,
                               partitioner);
        assert(inputs.count(value) &&
               "Input should be present after conversion");
      }
      auto inputInfo = inputs[value];
      nonPartitionInputs[value] = inputInfo;
      if (!inputArgs.count(inputInfo.tensor)) {
        inputArgs.insert({inputInfo.tensor, inputArgIndex++});
      }
    }
  }

  /**
   * @brief Collects the outputs of a given task partition.
   *
   * This function iterates over the outgoing edges of the specified partition
   * and collects the output values and their types. It ensures that all output
   * values have the same type and updates the result type accordingly.
   *
   * @param partition The task partition whose outputs are to be collected.
   * @param inputs The input map for the partition.
   * @param partitioner The graph partitioner used to manage the partition.
   * @param resultType The type of the result, which will be updated based on
   * the collected outputs.
   * @param bodyResults A vector to store the types of the body results.
   * @param nonPartitionOutputs A vector to store the output values that are not
   * part of the partition.
   */
  void collectTaskOutputs(SPNGraph &partition, InputMap &inputs,
                          GraphPartitioner &partitioner,
                          std::optional<Type> &resultType,
                          SmallVector<Type> &bodyResults,
                          SmallVector<Value> &nonPartitionOutputs) const {
    for (auto globalOutEdge : partitioner.edges_out(partition)) {
      auto value =
          boost::get(SPNEdge_Value(), partitioner.graph(), globalOutEdge);
      auto rType = performTypeConversion(value.getType());
      if (!resultType.has_value()) {
        resultType = rType;
      } else {
        assert(resultType.value() == rType &&
               "Multiple results with different types");
      }
      bodyResults.push_back(rType);
      nonPartitionOutputs.push_back(value);
    }
    assert(resultType.has_value() &&
           "Expecting at least one output from partition");
  }

  RankedTensorType createOutputType(Type resultType, size_t numResults) const {
    return RankedTensorType::get(
        {static_cast<long>(numResults), ShapedType::kDynamic}, resultType);
  }

  SmallVector<Value>
  getTaskInputs(const llvm::MapVector<Value, unsigned> &inputArgs) const {
    SmallVector<Value> taskInputs;
    for (auto &in : inputArgs) {
      taskInputs.push_back(in.first);
    }
    return taskInputs;
  }

  SPNTask createLoSPNTask(PatternRewriter &rewriter, Location loc,
                          RankedTensorType outputType,
                          const SmallVector<Value> &taskInputs,
                          unsigned batchSize) const {
    auto task =
        rewriter.create<SPNTask>(loc, outputType, taskInputs, batchSize);
    rewriter.modifyOpInPlace(task, [&task]() { task.addEntryBlock(); });
    return task;
  }

  /**
   * @brief Creates batch extracts for task inputs.
   *
   * This function processes the inputs for a given SPN task, creating batch
   * extracts for each input and storing the results in the body inputs vector.
   *
   * @param task The SPN task for which batch extracts are being created.
   * @param nonPartitionInputs A map of non-partitioned inputs.
   * @param inputArgs A map of input arguments with their corresponding indices.
   * @param inputIndices A map to store the indices of the inputs.
   * @param bodyInputs A vector to store the body inputs.
   * @param hasLogType An indexed map to indicate if the input has a log type.
   * @param rewriter The pattern rewriter used to create the batch extracts.
   * @param loc The location information for the created operations.
   */
  void createBatchExtractsForTaskInputs(
      SPNTask task, InputMap &nonPartitionInputs,
      const llvm::MapVector<Value, unsigned> &inputArgs,
      llvm::DenseMap<Value, unsigned> &inputIndices,
      SmallVector<Value> &bodyInputs, llvm::IndexedMap<bool> &hasLogType,
      PatternRewriter &rewriter, Location loc) const {
    unsigned bodyArgIndex = 0;
    hasLogType.grow(nonPartitionInputs.size());
    for (auto &in : nonPartitionInputs) {
      auto value = in.getFirst();
      auto inputInfo = in.getSecond();
      auto index = inputArgs.lookup(inputInfo.tensor);
      hasLogType[bodyArgIndex] = value.getType().isa<low::LogType>();
      inputIndices[value] = bodyArgIndex++;
      bool transposed = inputInfo.transposed();
      unsigned staticIndex =
          transposed ? inputInfo.rowIndex.value() : inputInfo.colIndex.value();
      auto extract = rewriter.create<SPNBatchExtract>(
          loc, performTypeConversion(value.getType()),
          task.getBody().front().getArgument(index), task.getBatchIndex(),
          staticIndex, rewriter.getBoolAttr(transposed));
      bodyInputs.push_back(extract);
    }
  }

  /**
   * @brief Creates an SPNBody task body with the specified parameters.
   *
   * This function creates an SPNBody task body, sets up the body block, adds
   * arguments to the block based on the input values and their log types, maps
   * the input values to the block arguments, and copies operations into the
   * task body.
   *
   * @param rewriter The PatternRewriter used to create and manipulate MLIR
   * operations.
   * @param loc The location information for the new SPNBody operation.
   * @param bodyResults A vector of types representing the result types of the
   * task body.
   * @param bodyInputs A vector of values representing the input values to the
   * task body.
   * @param hasLogType An IndexedMap indicating whether each input value has a
   * log type.
   * @param inputIndices A DenseMap mapping input values to their corresponding
   * indices.
   * @param partition The SPNGraph representing the partition of the task.
   * @param mapper The IRMapping used to map values between the original and the
   * new task body.
   * @return The created SPNBody task body.
   */
  SPNBody createTaskBody(PatternRewriter &rewriter, Location loc,
                         const SmallVector<Type> &bodyResults,
                         const SmallVector<Value> &bodyInputs,
                         llvm::IndexedMap<bool> &hasLogType,
                         llvm::DenseMap<Value, unsigned> &inputIndices,
                         SPNGraph &partition, IRMapping &mapper) const {
    auto body = rewriter.create<SPNBody>(loc, bodyResults, bodyInputs);
    auto bodyBlock = rewriter.createBlock(&body.getBody());
    rewriter.setInsertionPointToStart(bodyBlock);
    unsigned index = 0;
    for (auto &bodyIn : bodyInputs) {
      if (hasLogType[index++]) {
        bodyBlock->addArgument(
            low::LogType::get(getContext(), bodyIn.getType()),
            body.getBody().getLoc());
      } else {
        bodyBlock->addArgument(bodyIn.getType(), body.getBody().getLoc());
      }
    }
    for (auto remapped : inputIndices) {
      mapper.map(remapped.getFirst(),
                 bodyBlock->getArgument(remapped.getSecond()));
    }
    copyOperationsIntoTaskBody(partition, rewriter, mapper);
    return body;
  }

  void copyOperationsIntoTaskBody(SPNGraph &partition,
                                  PatternRewriter &rewriter,
                                  IRMapping &mapper) const {
    for (auto vertex : boost::make_iterator_range(boost::vertices(partition))) {
      auto *op = boost::get(SPNVertex_Operation(), partition, vertex);
      copyOperation(op, rewriter, mapper);
    }
  }

  /**
   * @brief Recursively copies an operation and its operands using a pattern
   * rewriter.
   *
   * This function ensures that all operands of the given operation are copied
   * before copying the operation itself. It uses an IRMapping to keep track of
   * already copied operations and operands to avoid duplication.
   *
   * @param op The operation to be copied.
   * @param rewriter The pattern rewriter used to clone the operation.
   * @param mapper The IRMapping that keeps track of the mapping between
   * original and copied operations.
   */
  void copyOperation(Operation *op, PatternRewriter &rewriter,
                     IRMapping &mapper) const {
    for (auto operand : op->getOperands()) {
      if (!mapper.contains(operand)) {
        assert(operand.getDefiningOp() &&
               "Operand is not the result of an operation");
        copyOperation(operand.getDefiningOp(), rewriter, mapper);
      }
    }
    if (!mapper.contains(op->getResult(0))) {
      rewriter.clone(*op, mapper);
    }
  }

  /**
   * @brief Updates the inputs and creates a yield operation in the given SPN
   * body.
   *
   * This function updates the input mappings and creates a yield operation
   * for the specified SPN task. It processes the non-partition outputs,
   * maps them to the corresponding values in the body, and updates the input
   * information accordingly.
   *
   * @param rewriter The pattern rewriter used to create the yield operation.
   * @param loc The location information for the yield operation.
   * @param body The SPN body where the yield operation will be created.
   * @param nonPartitionOutputs The outputs that are not partitioned.
   * @param inputs The input map to be updated with new input information.
   * @param task The SPN task for which the yield operation is created.
   * @param mapper The IR mapping used to map values from the original to the
   * new context.
   */
  void updateInputsAndCreateYield(PatternRewriter &rewriter, Location loc,
                                  SPNBody body,
                                  SmallVector<Value> &nonPartitionOutputs,
                                  InputMap &inputs, SPNTask task,
                                  IRMapping &mapper) const {
    SmallVector<Value> bodyYields;
    for (auto retVal : nonPartitionOutputs) {
      bodyYields.push_back(mapper.lookupOrNull(retVal));
      inputs[retVal] =
          InputInfo{task->getResult(0), bodyYields.size() - 1, std::nullopt};
    }
    rewriter.create<SPNYield>(loc, bodyYields);
  }

  /**
   * @brief Creates a batch collect operation and a return operation.
   *
   * This function creates an SPNBatchCollect operation using the provided
   * PatternRewriter, location, SPNTask, and SPNBody. It then creates an
   * SPNReturn operation with the result of the SPNBatchCollect operation.
   *
   * @param rewriter The PatternRewriter used to create the operations.
   * @param loc The location information for the new operations.
   * @param task The SPNTask containing the batch index.
   * @param body The SPNBody containing the results to be collected.
   */
  void createBatchCollectAndReturn(PatternRewriter &rewriter, Location loc,
                                   SPNTask task, SPNBody body) const {
    auto collect = rewriter.create<SPNBatchCollect>(loc, body->getResults(),
                                                    task.getBatchIndex(), true);
    rewriter.create<SPNReturn>(loc, collect.getResult());
  }

  /// Strip the LogType.
  /// \param type Type.
  /// \return The type or the base-type in case type is a LogType.
  Type performTypeConversion(Type type) const {
    if (type.isa<low::LogType>()) {
      return type.cast<low::LogType>().getBaseType();
    }
    return type;
  }

  TargetExecutionModel targetModel_;
  int maxTaskSize_;

  // True if the partitions should be scheduled.
  bool schedule_;

  // If enabled, each task gets only the part of the input tensor that it
  // actually uses. For this, each tasks input tensor is gathered from the
  // external tensor.
  bool decomposeTaskInputs_;
};

struct LoSPNTaskPartitioner
    : public impl::LoSPNTaskPartioningBase<LoSPNTaskPartitioner> {

public:
  using Base::Base;

protected:
  void runOnOperation() override {
    if (this->maxTaskSize > 0) {

      TargetExecutionModel targetModel;

      RewritePatternSet patterns(getOperation()->getContext());
      patterns.insert<PartitionTask>(getOperation()->getContext(), targetModel,
                                     this->maxTaskSize, this->schedule,
                                     this->decomposeTaskInputs);
      mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));
      if (failed(
              applyPatternsAndFoldGreedily(getOperation(), frozenPatterns))) {
        signalPassFailure();
      }
    }
  }
};

} // namespace low
} // namespace spn
} // namespace mlir