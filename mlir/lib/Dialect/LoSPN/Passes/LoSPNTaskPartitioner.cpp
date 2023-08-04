//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <llvm/ADT/IndexedMap.h>
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "LoSPNPassDetails.h"
#include "LoSPN/LoSPNPasses.h"
#include "LoSPN/LoSPNOps.h"
#include "LoSPN/LoSPNTypes.h"
#include "../Partitioning/GraphPartitioner.h"

namespace mlir {
  namespace spn {
    namespace low {

      ///
      /// Pattern matching a LoSPN task and splitting the task
      /// into multiple tasks by graph partitioning.
      class PartitionTask : public OpRewritePattern<low::SPNTask> {

      public:
        PartitionTask(MLIRContext* ctx, GraphPartitioner& part) :
            OpRewritePattern<low::SPNTask>(ctx, 1), partitioner(part) {}

        LogicalResult matchAndRewrite(SPNTask op, PatternRewriter& rewriter) const override {
          // All operations in the Task relevant for partitioning
          SmallVector<Operation*> nodes;
          // All operations with potentially no internal operands (in-degree 0)
          SmallPtrSet<Operation*, 10> inNodes;
          // All inputs considered external for the partitioning.
          SmallVector<Value> external;
          // Mapping from Value to a Tensor + index, either from an external
          // input of this task or internally from another partition.
          InputMap inputs;
          SmallVector<Value> taskResults;
          unsigned numNodes = 0;
          {
            // Map arguments of the entry block to external inputs of the task.
            llvm::DenseMap<mlir::BlockArgument, mlir::Value> externalTensors;
            unsigned tensorIndex = 0;
            bool initial = true;
            for (auto& blockArg : op.getBody()->getArguments()) {
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
                external.push_back(blockArg);
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
                // All users of the entry block args potentially do not have outside operands.
                for (auto* U : blockArg.getUsers()) {
                  inNodes.insert(U);
                }
              }
              body.body().walk([&](Operation* op) {
                if (isa<SPNYield>(op)) {
                  // SPNYield is not considered during partitioning, but the we need
                  // store the returned results to identify the new task producing the results.
                  for (auto resVal : op->getOperands()) {
                    taskResults.push_back(resVal);
                  }
                } else {
                  nodes.push_back(op);
                  if (op->hasTrait<OpTrait::ConstantLike>()) {
                    // Constant operations do not have an operand, so they
                    // should be used as seeds for the initial partitioning, too.
                    inNodes.insert(op);
                  } else {
                    ++numNodes;
                  }
                }
              });
            });
          }
          if (numNodes <= partitioner.getMaximumPartitionSize()) {
            // Do not partition a task if it is already smaller than the maximum size.
            return mlir::failure();
          }
          // Perform the actual partitioning.
          auto partition = partitioner.partitionGraph(nodes, inNodes, external);
          SmallVector<PartitionInfo> partitions;
          unsigned index = 0;
          for (auto& p : partition) {
            ++index;
            partitions.push_back(PartitionInfo{p.get(), false});
          }

          // Post-processing of constants: If a constant has a use in a different partition,
          // clone the constant to the other partition to avoid unnecessary edges crossing partitions.
          postprocessConstants(partitions, rewriter);
          // Handle the special case that a partition only contains constants.
          for (auto I = partitions.begin(); I != partitions.end();) {
            auto allConstant = llvm::all_of(*I->first, [](Operation* o) {
              return o->hasTrait<OpTrait::ConstantLike>();
            });
            if (allConstant) {
              I = partitions.erase(I);
            } else {
              ++I;
            }
          }

          // Create a new LoSPN task for each partition.
          for (auto& p : partitions) {
            createTaskForPartition(p, rewriter, op.getLoc(), op.batchSize(), inputs, partitions);
          }

          // Emit a remark with some information about the number of partitions etc.
          auto numPartitions = partitions.size();
          auto maxSize = partitioner.getMaximumPartitionSize();
          op->emitRemark() << "Split task into " << numPartitions << " partitions with a maximum size of " << maxSize;
          // Identify the task(s) producing the final result(s) of the original task and replace
          // the original task by the newly created tasks.
          SmallVector<Value> newResults;
          for (auto res : taskResults) {
            newResults.push_back(inputs.lookup(res).tensor);
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

        using PartitionInfo = std::pair<Partition*, bool>;

        void createTaskForPartition(PartitionInfo partition, PatternRewriter& rewriter, Location loc,
                                    unsigned batchSize,
                                    InputMap& inputs, llvm::ArrayRef<PartitionInfo> partitions) const {
          // First, collect all input values coming from either outside arguments of the original task or
          // from other partitions.
          InputMap nonPartitionInputs;
          llvm::MapVector<Value, unsigned> inputArgs;
          unsigned inputArgIndex = 1;
          for (auto* o : partition.first->hasExternalInputs()) {
            for (auto operand : o->getOperands()) {
              if (!partition.first->contains(operand.getDefiningOp())) {
                if (!inputs.count(operand)) {
                  // First, check the map for pre-existing entries.
                  // If no mapping is present, the input must be produced by another partition.
                  auto otherPartition = findPartition(operand, partitions);
                  assert(otherPartition.first && "Did not find partition producing this value");
                  // Convert the partition producing the input to a task first.
                  createTaskForPartition(otherPartition, rewriter, loc, batchSize, inputs, partitions);
                  // Input should be present after conversion.
                  assert(inputs.count(operand));
                }
                auto inputInfo = inputs[operand];
                nonPartitionInputs[operand] = inputInfo;
                // Remember which output from the outside will be provided by which argument to this task.
                if (!inputArgs.count(inputInfo.tensor)) {
                  inputArgs.insert({inputInfo.tensor, inputArgIndex++});
                }
              }
            }
          }
          // Collect information about which values this task will provide to other partitions.
          SmallVector<Value> nonPartitionOutputs;
          llvm::Optional<Type> resultType;
          SmallVector<Type> bodyResults;
          for (auto* o : partition.first->hasExternalOutputs()) {
            for (auto result : o->getResults()) {
              // Only add to nonPartitionOutputs if there's at least one user outside of the partition.
              auto hasExternalUser = llvm::any_of(result.getUsers(), [&partition](auto* U) {
                return !partition.first->contains(U);
              });
              if (hasExternalUser) {
                auto rType = performTypeConversion(result.getType());
                if (!resultType.hasValue()) {
                  resultType = rType;
                } else {
                  // Currently we assume that all results from one partition have the same type.
                  assert(resultType.getValue() == rType && "Multiple results with different types");
                }
                bodyResults.push_back(rType);
                nonPartitionOutputs.push_back(result);
              }
            }
          }
          assert(resultType.hasValue() && "Expecting at least one output from every partition");
          // All results of a partition are stored into one tensor (later on buffer).
          auto outputType = RankedTensorType::get({static_cast<long>(bodyResults.size()), -1},
                                                  resultType.getValue());
          // Add all input tensors as operands of the new task.
          SmallVector<Value> taskInputs;
          for (auto& in : inputArgs) {
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
          for (auto& in : nonPartitionInputs) {
            auto value = in.getFirst();
            auto inputInfo = in.getSecond();
            auto index = inputArgs[inputInfo.tensor];
            hasLogType[bodyArgIndex] = value.getType().isa<low::LogType>();
            // Remember which input value is associated with which input index for the body.
            inputIndices[value] = bodyArgIndex++;
            bool transposed = inputInfo.transposed();
            unsigned staticIndex = (transposed) ? inputInfo.rowIndex.getValue() : inputInfo.colIndex.getValue();
            auto extract = rewriter.create<SPNBatchExtract>(loc,
                                                            performTypeConversion(value.getType()),
                                                            taskBlock->getArgument(index),
                                                            task.getBatchIndex(), staticIndex,
                                                            rewriter.getBoolAttr(transposed));
            bodyInputs.push_back(extract);
          }
          auto body = rewriter.create<SPNBody>(loc, bodyResults, bodyInputs);
          auto restoreBody = rewriter.saveInsertionPoint();
          auto bodyBlock = rewriter.createBlock(&body.body());
          auto index = 0;
          // Add an block argument for each external input. The block arg corresponds to the
          // batch extract extracing the value from the input tensor.
          for (auto& bodyIn : bodyInputs) {
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
          for (auto operation : *partition.first) {
            copyOperation(operation, rewriter, mapper);
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


        void copyOperation(Operation* op, PatternRewriter& rewriter, BlockAndValueMapping& mapper) const {
          for (auto operand : op->getOperands()) {
            if (!mapper.contains(operand)) {
              // Copy definition first to ensure legality of def-use-chains
              copyOperation(operand.getDefiningOp(), rewriter, mapper);
            }
          }
          // Make sure we do not copy an operation twice, if it has previously
          // been copied as an operand of another operation.
          if (!mapper.contains(op->getResult(0))) {
            (void) rewriter.clone(*op, mapper);
          }
        }

        /// Find which partition contains the operation producing the given value.
        /// \param input The produced Value.
        /// \param partitions List of partitions.
        /// \return Information about the partition containing the operation/value or a
        ///         nullptr wrapped in the information if no partition could be found.
        PartitionInfo findPartition(Value input, llvm::ArrayRef<PartitionInfo> partitions) const {
          for (auto& p : partitions) {
            if (p.second) {
              // This partition has already been converted and all outputs should be present in the map.
              continue;
            }
            for (auto* o : p.first->hasExternalOutputs()) {
              if (o == input.getDefiningOp()) {
                return p;
              }
            }
          }
          return PartitionInfo{nullptr, false};
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

        void postprocessConstants(llvm::ArrayRef<PartitionInfo> partitions, PatternRewriter& rewriter) const {
          for (auto& p : partitions) {
            for (auto* out : p.first->hasExternalOutputs()) {
              if (out->hasTrait<OpTrait::ConstantLike>()) {
                assert(out->getNumResults() == 1);
                for (auto& use : out->getUses()) {
                  if (!p.first->contains(use.getOwner())) {
                    // This user is located in another partition.
                    // Find the partition of the using operation.
                    auto otherPart = findContainingPartition(use.getOwner(), partitions);
                    assert(otherPart.first);
                    // Clone the constant right before the using operation and add it to the same partition.
                    auto restore = rewriter.saveInsertionPoint();
                    rewriter.setInsertionPoint(use.getOwner());
                    auto clonedOut = rewriter.clone(*out);
                    otherPart.first->addNode(clonedOut);
                    use.set(clonedOut->getResult(0));
                    rewriter.restoreInsertionPoint(restore);
                    p.first->invalidateExternal();
                  }
                }
              }
            }
          }
        }

        /// Find the partition containing the specified operation.
        /// \param op Operation.
        /// \param partitions List of partitions.
        /// \return Information about the partition containing the operation or a
        //          nullptr wrapped in the information if no partition could be found.
        PartitionInfo findContainingPartition(Operation* op, llvm::ArrayRef<PartitionInfo> partitions) const {
          for (auto& p : partitions) {
            if (p.first->contains(op)) {
              return p;
            }
          }
          return PartitionInfo{nullptr, false};
        }

        GraphPartitioner& partitioner;

      };

      struct LoSPNTaskPartitioner : public LoSPNTaskPartioningBase<LoSPNTaskPartitioner> {

      public:

        LoSPNTaskPartitioner() = default;

        explicit LoSPNTaskPartitioner(int maxTaskSize) {
          this->maxTaskSize = maxTaskSize;
        }

      protected:

        void runOnOperation() override {
          if (this->maxTaskSize.getValue() > 0) {
            GraphPartitioner partitioner{this->maxTaskSize.getValue(), SimpleMoveHeuristic::create};
            RewritePatternSet patterns(getOperation()->getContext());
            patterns.insert<PartitionTask>(getOperation()->getContext(), partitioner);
            mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));
            if (failed(applyPatternsAndFoldGreedily(getOperation(), frozenPatterns))) {
              signalPassFailure();
            }
          }
        }

      };

    }
  }
}

std::unique_ptr<mlir::OperationPass<mlir::spn::low::SPNKernel>> mlir::spn::low::createLoSPNPartitionerPass(int maxTaskSize) {
  return std::make_unique<LoSPNTaskPartitioner>(maxTaskSize);
}