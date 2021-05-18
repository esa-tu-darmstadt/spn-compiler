//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include <mlir/Rewrite/FrozenRewritePatternSet.h>
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

      class PartitionTask : public OpRewritePattern<low::SPNTask> {

      public:
        PartitionTask(MLIRContext* ctx, GraphPartitioner& part) :
            OpRewritePattern<low::SPNTask>(ctx, 1), partitioner(part) {}

        LogicalResult matchAndRewrite(SPNTask op, PatternRewriter& rewriter) const override {
          // All operations in the Task relevant for partitioning
          SmallVector<Operation*> nodes;
          // All operations with no internal operands (in-degree 0)
          SmallVector<Operation*> inNodes;
          // All inputs considered external for the partitioning.
          SmallVector<Value> external;
          // Mapping from Value to a Tensor + index, either from an external
          // input of this task or internally from another partition.
          InputMap inputs;
          SmallVector<Value> taskResults;
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
                auto collect = cast<SPNBatchExtract>(bodyOp.getDefiningOp());
                assert(collect.input().isa<BlockArgument>());
                // Get the input tensor of the BatchExtract
                auto tensorArg = collect.input();
                // Match the input tensor of the BatchExtract
                // (which should be a block argument of the Task's entry block)
                // to the external operand of the task.
                assert(externalTensors.count(tensorArg.cast<BlockArgument>()));
                auto externalTensor = externalTensors[tensorArg.cast<BlockArgument>()];
                inputs[blockArg] = InputInfo{externalTensor, collect.sampleIndex()};
                // All users of the entry block args potentially do not have outside operands.
                for (auto* U : blockArg.getUsers()) {
                  inNodes.push_back(U);
                }
              }
              body.body().walk([&](Operation* op) {
                if (isa<SPNYield>(op)) {
                  for (auto resVal : op->getOperands()) {
                    taskResults.push_back(resVal);
                  }
                } else {
                  nodes.push_back(op);
                  if (isa<SPNConstant>(op)) {
                    // Constant operations do not have an operand, so they
                    // should be used as seeds for the initial partitioning, too.
                    inNodes.push_back(op);
                  }
                }
              });
            });
          }
          if (nodes.size() < 8) {
            return mlir::failure();
          }
          // Perform the actual partitioning.
          auto partition = partitioner.partitionGraph(nodes, inNodes, external);
          SmallVector<PartitionInfo> partitions;
          unsigned index = 0;
          for (auto& p : partition) {
            llvm::dbgs() << "Partition " << index << ":\n";
            for (auto o : *p) {
              o->dump();
            }
            ++index;
            partitions.push_back(PartitionInfo{p.get(), false});
          }
          // Keep track of operations moved to new, partitioned task.
          SmallVector<Operation*> movedOperations;
          for (auto& p : partitions) {
            createTaskForPartition(p, rewriter, op.getLoc(), op.batchSize(), inputs, partitions, movedOperations);
          }
          SmallVector<Value> newResults;
          for (auto res : taskResults) {
            newResults.push_back(inputs.lookup(res).first);
          }
          // Delete all the operations moved to new, partitioned tasks from the original
          // task to avoid errors involving deleted, but used operations.
          for (auto* movedOp : llvm::reverse(movedOperations)) {
            if (!movedOp->getUsers().empty()) {
              for (auto* U : movedOp->getUsers()) {
                // SPNYield can savely be deleted, the body will stop existing after
                // the original task is also deleted.
                if (isa<SPNYield>(U)) {
                  rewriter.eraseOp(U);
                }
              }
            }
            rewriter.eraseOp(movedOp);
          }
          //assert(false);
          rewriter.replaceOp(op, newResults);
          return mlir::success();
        }

      private:

        using InputInfo = std::pair<mlir::Value, unsigned>;

        using InputMap = llvm::DenseMap<mlir::Value, InputInfo>;

        using PartitionInfo = std::pair<Partition*, bool>;

        void createTaskForPartition(PartitionInfo partition, PatternRewriter& rewriter, Location loc,
                                    unsigned batchSize,
                                    InputMap& inputs, llvm::ArrayRef<PartitionInfo> partitions,
                                    SmallVectorImpl<Operation*>& movedOperations) const {
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
                  // Convert the partition producing the input to a task first.
                  createTaskForPartition(otherPartition, rewriter, loc, batchSize, inputs,
                                         partitions, movedOperations);
                  // Input should be present after conversion.
                  assert(inputs.count(operand));
                }
                auto inputInfo = inputs[operand];
                nonPartitionInputs[operand] = inputInfo;
                if (!inputArgs.count(inputInfo.first)) {
                  inputArgs[inputInfo.first] = inputArgIndex++;
                }
              }
            }
          }
          // Collect information about which values this task will provide to other partitions.
          SmallVector<Value> nonPartitionOutputs;
          SmallVector<Type> resultTypes;
          SmallVector<Type> bodyResults;
          for (auto* o : partition.first->hasExternalOutputs()) {
            for (auto result : o->getResults()) {
              // Only add to nonPartitionOutputs if there's at least one user outside of the partition.
              for (auto* U : result.getUsers()) {
                if (!partition.first->contains(U)) {
                  nonPartitionOutputs.push_back(result);
                  auto resultType = performTypeConversion(result.getType());
                  resultTypes.push_back(RankedTensorType::get({-1}, resultType));
                  bodyResults.push_back(resultType);
                  break;
                }
              }
            }
          }
          SmallVector<Value> taskInputs;
          for (auto& in : inputArgs) {
            taskInputs.push_back(in.first);
          }
          auto task = rewriter.create<SPNTask>(loc, resultTypes, taskInputs, batchSize);
          auto restore = rewriter.saveInsertionPoint();
          auto taskBlock = task.addEntryBlock();
          rewriter.setInsertionPointToStart(taskBlock);
          llvm::DenseMap<Value, unsigned> inputIndices;
          SmallVector<Value> bodyInputs;
          unsigned bodyArgIndex = 0;
          bool hasLogType[nonPartitionInputs.size()];
          for (auto& in : nonPartitionInputs) {
            auto value = in.getFirst();
            auto inputInfo = in.getSecond();
            auto index = inputArgs[inputInfo.first];
            hasLogType[bodyArgIndex] = value.getType().isa<low::LogType>();
            inputIndices[value] = bodyArgIndex++;
            auto extract = rewriter.create<SPNBatchExtract>(loc,
                                                            performTypeConversion(value.getType()),
                                                            taskBlock->getArgument(index),
                                                            task.getBatchIndex(), inputInfo.second);
            bodyInputs.push_back(extract);
          }
          auto body = rewriter.create<SPNBody>(loc, bodyResults, bodyInputs);
          auto restoreBody = rewriter.saveInsertionPoint();
          auto bodyBlock = rewriter.createBlock(&body.body());
          auto index = 0;
          for (auto& bodyIn : bodyInputs) {
            if (hasLogType[index++]) {
              bodyBlock->addArgument(low::LogType::get(bodyIn.getType()));
            } else {
              bodyBlock->addArgument(bodyIn.getType());
            }
          }
          BlockAndValueMapping mapper;
          for (auto remapped : inputIndices) {
            mapper.map(remapped.getFirst(), bodyBlock->getArgument(remapped.second));
          }
          for (auto operation : *partition.first) {
            copyOperation(operation, rewriter, mapper, movedOperations);
          }
          SmallVector<Value> bodyYields;
          unsigned resultIndex = 0;
          for (auto retVal : nonPartitionOutputs) {
            bodyYields.push_back(mapper.lookupOrNull(retVal));
            inputs[retVal] = InputInfo{task->getResult(resultIndex++), 0};
          }
          rewriter.create<SPNYield>(loc, bodyYields);
          rewriter.restoreInsertionPoint(restoreBody);
          SmallVector<Value> taskReturns;
          for (auto r : body.getResults()) {
            auto rType = RankedTensorType::get({-1}, r.getType());
            auto collect = rewriter.create<SPNBatchCollect>(loc, rType, r, task.getBatchIndex());
            taskReturns.push_back(collect.getResult(0));
          }
          rewriter.create<SPNReturn>(loc, taskReturns);
          rewriter.restoreInsertionPoint(restore);
        }

        void copyOperation(Operation* op, PatternRewriter& rewriter, BlockAndValueMapping& mapper,
                           SmallVectorImpl<Operation*>& moved) const {
          for (auto operand : op->getOperands()) {
            if (!mapper.contains(operand)) {
              // Copy definition first to ensure legality of def-use-chains
              copyOperation(operand.getDefiningOp(), rewriter, mapper, moved);
            }
          }
          (void) rewriter.clone(*op, mapper);
          moved.push_back(op);
        }

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

        Type performTypeConversion(Type type) const {
          if (type.isa<low::LogType>()) {
            return type.cast<low::LogType>().getBaseType();
          }
          return type;
        }

        GraphPartitioner& partitioner;

      };

      struct LoSPNTaskPartitioner : LoSPNTaskPartioningBase<LoSPNTaskPartitioner> {

      protected:

        void runOnOperation() override {
          GraphPartitioner partitioner{};
          RewritePatternSet patterns(getOperation()->getContext());
          patterns.insert<PartitionTask>(getOperation()->getContext(), partitioner);
          mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));
          applyPatternsAndFoldGreedily(getOperation(), frozenPatterns);
        }

      };

    }
  }
}

std::unique_ptr<mlir::OperationPass<mlir::spn::low::SPNKernel>> mlir::spn::low::createLoSPNPartitionerPass() {
  return std::make_unique<LoSPNTaskPartitioner>();
}