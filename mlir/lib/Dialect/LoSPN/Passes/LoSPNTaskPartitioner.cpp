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
#include "LoSPN/LoSPNTypes.h"
#include "LoSPNPassDetails.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Use.h"
#include <boost/graph/properties.hpp>
#include <boost/graph/subgraph.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/iterator_range_core.hpp>
#include <llvm/ADT/IndexedMap.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>

namespace mlir {
namespace spn {
namespace low {

using namespace mlir::spn::low::partitioning;

///
/// Pattern matching a LoSPN task and splitting the task
/// into multiple tasks by graph partitioning.
class PartitionTask : public OpRewritePattern<low::SPNTask> {

public:
  PartitionTask(MLIRContext *ctx, int maxTaskSize)
      : OpRewritePattern<low::SPNTask>(ctx, 1), maxTaskSize_(maxTaskSize) {}

  LogicalResult matchAndRewrite(SPNTask op, PatternRewriter &rewriter) const override {
    // All operations in the Task relevant for partitioning
    SmallVector<Operation *> nodes;
    // Mapping from Value to a Tensor + index, either from an external
    // input of this task or internally from another partition.
    InputMap inputs;
    SmallVector<Operation *> taskTerminators;
    unsigned numNodes = 0;
    {
      // Map arguments of the entry block to external inputs of the task.
      llvm::DenseMap<mlir::BlockArgument, mlir::Value> externalTensors;
      unsigned tensorIndex = 0;
      bool initial = true;
      for (auto &blockArg : op.getBody()->getArguments()) {
        if (initial) {
          // The first block argument is the batch index, skip it.
          initial = false;
          continue;
        }
        externalTensors[blockArg] = op->getOperand(tensorIndex++);
      }
      op->walk([&](SPNBody body) {
        unsigned inIndex = 0;
        for (auto blockArg : body.getBody()->getArguments()) {
          // Detect the BatchExtract producing this block arg:
          auto bodyOp = body->getOperand(inIndex++);
          assert(isa<SPNBatchExtract>(bodyOp.getDefiningOp()));
          auto extract = cast<SPNBatchExtract>(bodyOp.getDefiningOp());
          assert(extract.input().isa<BlockArgument>());
          // Get the input tensor of the BatchExtract
          auto tensorArg = extract.input();
          // Match the input tensor of the BatchExtract
          // (which should be a block argument of the Task's entry block)
          // to the external operand of the task.
          assert(externalTensors.count(tensorArg.cast<BlockArgument>()));
          auto externalTensor = externalTensors[tensorArg.cast<BlockArgument>()];
          inputs[blockArg] = InputInfo{externalTensor, llvm::None, extract.staticIndex()};
        }
        body.body().walk([&](Operation *op) {
          if (isa<SPNYield>(op)) {
            // Store terminators
            taskTerminators.push_back(op);
          } else {
            nodes.push_back(op);
            if (op->hasTrait<OpTrait::ConstantLike>()) {
              // Constant operations do not have an operand, so they
              // should be used as seeds for the initial partitioning, too.
            } else {
              ++numNodes;
            }
          }
        });
      });
    }
    if ((int)numNodes <= maxTaskSize_) {
      // Do not partition a task if it is already smaller than the maximum size.
      return mlir::failure();
    }

    // Perform the actual partitioning
    GraphPartitioner partitioner(taskTerminators, maxTaskSize_);
    partitioner.clusterGraph();
    partitioner.scheduleGraphForBSP();

    // Post-processing of constants: If a constant has a use in a different partition,
    // clone the constant to the other partition to avoid unnecessary edges crossing partitions.
    postprocessConstants(partitioner, rewriter);

    // Create a new LoSPN task for each cluster.
    for (auto &cluster : partitioner.clusters()) {
      // Skip clusters that only contain constants.
      auto vertices = boost::vertices(cluster);
      if (std::all_of(vertices.first, vertices.second,
                      [&cluster](auto vertex) { return boost::get(SPNVertex_IsConstant(), cluster, vertex); }))
        continue;

      // Create a new task for this cluster.
      createTaskForPartition(cluster, rewriter, op.getLoc(), op.batchSize(), inputs, partitioner);
    }

    // Emit a remark with some information about the number of partitions etc.
    auto numPartitions = partitioner.numClusters();
    auto maxSize = partitioner.getMaximumClusterSize();
    op->emitRemark() << "Split task into " << numPartitions << " partitions with a maximum size of " << maxSize;
    // Identify the task(s) producing the final result(s) of the original task and replace
    // the original task by the newly created tasks.
    SmallVector<Value> newResults;
    for (auto op : taskTerminators) {
      for (auto resVal : op->getOperands()) {
        newResults.push_back(inputs.lookup(resVal).tensor);
      }
    }
    rewriter.replaceOp(op, newResults);
    return mlir::success();
  }

private:
  struct InputInfo {
    Value tensor;
    llvm::Optional<unsigned> rowIndex;
    llvm::Optional<unsigned> colIndex;

    bool transposed() const {
      assert(rowIndex.hasValue() ^ colIndex.hasValue());
      return rowIndex.hasValue();
    }
  };

  using InputMap = llvm::DenseMap<mlir::Value, InputInfo>;
  using InputMap = llvm::DenseMap<mlir::Value, InputInfo>;

  void createTaskForPartition(SPNGraph &partition, PatternRewriter &rewriter, Location loc, unsigned batchSize,
                              InputMap &inputs, GraphPartitioner &partitioner) const {
    // First, collect all input values coming from outside arguments of the original task
    InputMap nonPartitionInputs;
    llvm::MapVector<Value, unsigned> inputArgs;
    unsigned inputArgIndex = 1;
    for (auto vertex : boost::make_iterator_range(boost::vertices(partition))) {
      // Check if this vertex uses an input argument of the original task.
      if (boost::get(SPNVertex_UsesInput(), partition, vertex)) {
        // Get the underlying operation of the vertex.
        auto *op = boost::get(SPNVertex_Operation(), partition, vertex);
        for (auto operands : op->getOperands()) {
          assert(inputs.count(operands) && "External input are expected to be present in the input map");
          auto inputInfo = inputs[operands];
          nonPartitionInputs[operands] = inputInfo;
          // Remember which output from the outside will be provided by which argument to this task.
          if (!inputArgs.count(inputInfo.tensor)) {
            inputArgs.insert({inputInfo.tensor, inputArgIndex++});
          }
        }
      }
    }
    // First, collect all input values coming from other partitions.
    for (auto globalInEdge : partitioner.edges_in(partition)) {
      Value value = boost::get(SPNEdge_Value(), partitioner.graph(), globalInEdge);
      // First, check the map for pre-existing entries.
      if (!inputs.count(value)) {
        // If no mapping is present, the input must be produced by another partition.
        auto globalVertexFrom = boost::source(globalInEdge, partitioner.graph());
        auto otherPartition = find_cluster(globalVertexFrom, partitioner.graph());
        // Convert the partition producing the input to a task first.
        createTaskForPartition(otherPartition, rewriter, loc, batchSize, inputs, partitioner);
        // Input should be present after conversion.
        assert(inputs.count(value));
      }
      auto inputInfo = inputs[value];
      nonPartitionInputs[value] = inputInfo;
      // Remember which output from the outside will be provided by which argument to this task.
      if (!inputArgs.count(inputInfo.tensor)) {
        inputArgs.insert({inputInfo.tensor, inputArgIndex++});
      }
    }
    // Collect information about which values this task will provide to other partitions.
    SmallVector<Value> nonPartitionOutputs;
    llvm::Optional<Type> resultType;
    SmallVector<Type> bodyResults;
    for (auto globalOutEdge : partitioner.edges_out(partition)) {
      auto value = boost::get(SPNEdge_Value(), partitioner.graph(), globalOutEdge);
      auto rType = performTypeConversion(value.getType());
      if (!resultType.hasValue()) {
        resultType = rType;
      } else {
        // Currently we assume that all results from one partition have the same type.
        assert(resultType.getValue() == rType && "Multiple results with different types");
      }
      bodyResults.push_back(rType);
      nonPartitionOutputs.push_back(value);
    }

    assert(resultType.hasValue() && "Expecting at least one output from every partition");
    // All results of a partition are stored into one tensor (later on buffer).
    auto outputType = RankedTensorType::get({static_cast<long>(bodyResults.size()), -1}, resultType.getValue());
    // Add all input tensors as operands of the new task.
    SmallVector<Value> taskInputs;
    for (auto &in : inputArgs) {
      taskInputs.push_back(in.first);
    }
    // Create the actual LoSPN task.
    auto task = rewriter.create<SPNTask>(loc, outputType, taskInputs, batchSize);
    auto restore = rewriter.saveInsertionPoint();
    auto taskBlock = task.addEntryBlock();
    // Create a batch extract for each tensor argument of the new task.
    rewriter.setInsertionPointToStart(taskBlock);
    llvm::DenseMap<Value, unsigned> inputIndices;
    SmallVector<Value> bodyInputs;
    unsigned bodyArgIndex = 0;
    llvm::IndexedMap<bool> hasLogType;
    hasLogType.grow(nonPartitionInputs.size());
    for (auto &in : nonPartitionInputs) {
      auto value = in.getFirst();
      auto inputInfo = in.getSecond();
      auto index = inputArgs[inputInfo.tensor];
      hasLogType[bodyArgIndex] = value.getType().isa<low::LogType>();
      // Remember which input value is associated with which input index for the body.
      inputIndices[value] = bodyArgIndex++;
      bool transposed = inputInfo.transposed();
      unsigned staticIndex = (transposed) ? inputInfo.rowIndex.getValue() : inputInfo.colIndex.getValue();
      auto extract =
          rewriter.create<SPNBatchExtract>(loc, performTypeConversion(value.getType()), taskBlock->getArgument(index),
                                           task.getBatchIndex(), staticIndex, rewriter.getBoolAttr(transposed));
      bodyInputs.push_back(extract);
    }
    auto body = rewriter.create<SPNBody>(loc, bodyResults, bodyInputs);
    auto restoreBody = rewriter.saveInsertionPoint();
    auto bodyBlock = rewriter.createBlock(&body.body());
    auto index = 0;
    // Add an block argument for each external input. The block arg corresponds to the
    // batch extract extracing the value from the input tensor.
    for (auto &bodyIn : bodyInputs) {
      if (hasLogType[index++]) {
        bodyBlock->addArgument(low::LogType::get(bodyIn.getType()));
      } else {
        bodyBlock->addArgument(bodyIn.getType());
      }
    }
    // Populate a mapping from external Value to block argument.
    BlockAndValueMapping mapper;
    for (auto remapped : inputIndices) {
      mapper.map(remapped.getFirst(), bodyBlock->getArgument(remapped.second));
    }
    // Copy the operations in this partition from the original task to the new task.
    for (auto vertex : boost::make_iterator_range(boost::vertices(partition))) {
      auto *op = boost::get(SPNVertex_Operation(), partition, vertex);
      copyOperation(op, rewriter, mapper);
    }
    SmallVector<Value> bodyYields;
    unsigned resultIndex = 0;
    // Create a SPNYield with all results at the end of the body.
    for (auto retVal : nonPartitionOutputs) {
      bodyYields.push_back(mapper.lookupOrNull(retVal));
      inputs[retVal] = InputInfo{task->getResult(0), resultIndex++, llvm::None};
    }
    rewriter.create<SPNYield>(loc, bodyYields);
    rewriter.restoreInsertionPoint(restoreBody);
    // Create a SPNBatchCollect collecting all scalar results into a single tensor.
    auto collect = rewriter.create<SPNBatchCollect>(loc, body->getResults(), task.getBatchIndex(), true);
    // Create a Return at the end of the task, returning all results as tensors.
    rewriter.create<SPNReturn>(loc, collect.getResult());
    rewriter.restoreInsertionPoint(restore);
  }

  void copyOperation(Operation *op, PatternRewriter &rewriter, BlockAndValueMapping &mapper) const {
    for (auto operand : op->getOperands()) {
      if (!mapper.contains(operand)) {
        // Copy definition first to ensure legality of def-use-chains

        // We expect the operand to be mapped if it not the result of an operation. Especially, all block arguments,
        // i.e., external inputs, should be already mapped.
        assert(operand.getDefiningOp() && "This operand is not the result of an operation but it is not mapped yet");
        copyOperation(operand.getDefiningOp(), rewriter, mapper);
      }
    }
    // Make sure we do not copy an operation twice, if it has previously
    // been copied as an operand of another operation.
    if (!mapper.contains(op->getResult(0))) {
      (void)rewriter.clone(*op, mapper);
    }
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

  void postprocessConstants(GraphPartitioner &partitioner, PatternRewriter &rewriter) const {
    for (auto &cluster : partitioner.clusters()) {
      for (auto globalOutEdge : partitioner.edges_out(cluster)) {
        auto globalVertexFrom = boost::source(globalOutEdge, partitioner.graph());
        if (boost::get(SPNVertex_IsConstant(), partitioner.graph(), globalVertexFrom)) {
          assert(boost::get(SPNVertex_Operation(), partitioner.graph(), globalVertexFrom)->getNumResults() == 1);
          // This constant is used by another partition.
          // Find the partition that uses the constant
          auto globalVertexTo = boost::target(globalOutEdge, partitioner.graph());
          auto otherPart = find_cluster(globalVertexTo, partitioner.graph());

          // Clone the constant right before the using operation and add it to the same partition.
          auto restore = rewriter.saveInsertionPoint();

          // Get the constant operation and the operation thats using it
          Value value = boost::get(SPNEdge_Value(), partitioner.graph(), globalOutEdge);
          Operation *constOperation = boost::get(SPNVertex_Operation(), partitioner.graph(), globalVertexFrom);
          Operation *usingOperation = boost::get(SPNVertex_Operation(), partitioner.graph(), globalVertexTo);

          rewriter.setInsertionPoint(usingOperation);
          auto clonedOut = rewriter.clone(*constOperation);

          // Add the cloned constant to the partition
          auto globalClonedConstant = add_vertex(otherPart, clonedOut);
          auto localClonedConstant = boost::add_vertex(globalClonedConstant, otherPart);

          // Add the edge from the cloned constant to the using operation in the other partition
          auto localVertexTo = otherPart.global_to_local(globalVertexTo);
          add_edge(localClonedConstant, localVertexTo, otherPart, clonedOut->getResult(0));

          usingOperation->replaceUsesOfWith(value, clonedOut->getResult(0));
          rewriter.restoreInsertionPoint(restore);
        }
      }
    }
  }

  int maxTaskSize_;
};

struct LoSPNTaskPartitioner : public LoSPNTaskPartioningBase<LoSPNTaskPartitioner> {

public:
  LoSPNTaskPartitioner() = default;

  explicit LoSPNTaskPartitioner(int maxTaskSize) { this->maxTaskSize = maxTaskSize; }

protected:
  void runOnOperation() override {
    if (this->maxTaskSize.getValue() > 0) {
      RewritePatternSet patterns(getOperation()->getContext());
      patterns.insert<PartitionTask>(getOperation()->getContext(), this->maxTaskSize.getValue());
      mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));
      if (failed(applyPatternsAndFoldGreedily(getOperation(), frozenPatterns))) {
        signalPassFailure();
      }
    }
  }
};

} // namespace low
} // namespace spn
} // namespace mlir

std::unique_ptr<mlir::OperationPass<mlir::spn::low::SPNKernel>>
mlir::spn::low::createLoSPNPartitionerPass(int maxTaskSize) {
  return std::make_unique<LoSPNTaskPartitioner>(maxTaskSize);
}