//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPU/Vectorization/SLP/SLPFunctionTransformer.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPUtil.h"
#include "LoSPN/LoSPNOps.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::low;
using namespace mlir::spn::low::slp;

SLPFunctionTransformer::SLPFunctionTransformer(std::unique_ptr<SLPNode>&& graph, MLIRContext* context) : root{
    std::move(graph)}, builder{context}, typeConverter{static_cast<unsigned int>(root->numLanes())} {

}


// Anonymous namespace for helper functions.
namespace {

  mlir::Value extractMemRefOperand(mlir::Operation* op, mlir::OpBuilder& builder) {
    if (auto batchRead = llvm::dyn_cast<mlir::spn::low::SPNBatchRead>(op)) {
      builder.setInsertionPoint(op);
      auto const& memRef = batchRead.batchMem();
      auto const& batchOffset = batchRead.batchIndex();
      auto const& sampleOffset =
          builder.create<mlir::ConstantOp>(op->getLoc(), builder.getIndexAttr(batchRead.sampleIndex()));
      mlir::ValueRange indices{batchOffset, sampleOffset->getResult(0)};
      return builder.create<mlir::LoadOp>(op->getLoc(), memRef, indices);
    }
    assert(false);
  }

  void eraseFromFunction(SLPNode const& root) {
    for (size_t lane = 0; lane < root.numLanes(); ++lane) {
      for (size_t i = 0; i < root.numVectors(); ++i) {
        root.getOperation(lane, i)->remove();
      }
    }
    for (auto const& operand : root.getOperands()) {
      eraseFromFunction(*operand);
    }
  }
}

void SLPFunctionTransformer::transform() {
  transform(root.get(), 0);
  eraseFromFunction(*root);
  // TODO: create remaining use extractions
}

Value SLPFunctionTransformer::transform(SLPNode* node, size_t vectorIndex) {
  node->dump();
  llvm::dbgs() << "\n";
  if (std::uintptr_t(node->getOperation(0, 0)) == 0x562d7a63bad8 || dyn_cast<SPNBatchRead>(node->getOperation(0, 0))) {
    llvm::dbgs() << "\n";
  }

  vectorsDone[node]++;

  auto vectorType = typeConverter.convertType(node->getResultType());

  auto const& vectorizableOps = node->getVector(vectorIndex);
  vectorizedOps.insert(vectorizableOps.begin(), vectorizableOps.end());
  auto const& firstOp = vectorizableOps.front();
  auto const& loc = firstOp->getLoc();

  if (areBroadcastable(vectorizableOps)) {
    builder.setInsertionPoint(firstOp);
    auto value = firstOp->getResult(0);
    auto vectorOp = builder.create<vector::BroadcastOp>(loc, vectorType, value);
    return applyCreation(node, vectorIndex, vectorOp);
  }

  if (areConsecutiveLoads(vectorizableOps)) {
    builder.setInsertionPoint(firstOp);
    auto batchRead = static_cast<SPNBatchRead>(firstOp);
    auto const& base = batchRead.batchMem();
    auto const& batchIndex = batchRead.batchIndex();
    auto const& offset = builder.create<ConstantOp>(loc, builder.getIndexAttr(batchRead.sampleIndex())).getResult();
    auto indices = ValueRange{batchIndex, offset};
    auto vectorOp = builder.create<vector::LoadOp>(loc, vectorType, base, indices);
    return applyCreation(node, vectorIndex, vectorOp);
  }

  if (canBeGathered(vectorizableOps)) {
    builder.setInsertionPoint(firstOp);
    std::vector<Value> elements;
    for (auto const& op : vectorizableOps) {
      auto batchRead = static_cast<SPNBatchRead>(op);
      auto const& base = batchRead.batchMem();
      auto const& batchIndex = batchRead.batchIndex();
      unsigned sampleIndex = batchRead.sampleIndex();
      auto const& memRefTuple = std::make_tuple(base, batchIndex, sampleIndex);
      if (memRefLoads.count(memRefTuple)) {
        elements.emplace_back(memRefLoads[memRefTuple]);
      } else {
        auto const& offset = builder.create<ConstantOp>(loc, builder.getIndexAttr(sampleIndex)).getResult();
        auto indices = ValueRange{batchIndex, offset};
        Value element = builder.create<LoadOp>(loc, base, indices);
        memRefLoads[memRefTuple] = element;
        elements.emplace_back(element);
      }
    }
    Operation* vectorOp = builder.create<vector::BroadcastOp>(loc, vectorType, elements[0]);
    for (size_t i = 1; i < vectorizableOps.size(); ++i) {
      vectorOp = builder.create<vector::InsertElementOp>(loc, elements[i], vectorOp->getResult(0), i);
    }
    vectorOp->getBlock()->dump();
    return applyCreation(node, vectorIndex, vectorOp);
  }

  if (node->isUniform() && firstOp->getNumOperands() > 0) {

    llvm::SmallVector<Value, 2> operands;

    for (size_t i = 0; i < firstOp->getNumOperands(); ++i) {

      if (node->numOperands() == 0) {
        size_t nextIndex = vectorsDone[node];
        if (nextIndex < node->numVectors()) {
          operands.emplace_back(transform(node, nextIndex));
        } else {
          auto nodeInput = node->getNodeInput(nodeInputsDone[node]++);
          nodeInput.dump();
          if (nodeInput.getType().isa<MemRefType>()) {
            nodeInput = extractMemRefOperand(vectorizableOps[0], builder);
            nodeInput.dump();
          }
          builder.setInsertionPointAfterValue(nodeInput);
          Operation* vectorOp = builder.create<vector::BroadcastOp>(loc, vectorType, nodeInput);
          for (size_t lane = 1; lane < node->numLanes(); ++lane) {
            nodeInput = node->getNodeInput(nodeInputsDone[node]++);
            nodeInput.dump();
            if (nodeInput.getType().isa<MemRefType>()) {
              nodeInput = extractMemRefOperand(vectorizableOps[lane], builder);
              nodeInput.dump();
            }
            builder.setInsertionPointAfterValue(nodeInput);
            vectorOp = builder.create<vector::InsertElementOp>(vectorOp->getLoc(),
                                                               nodeInput,
                                                               vectorOp->getResult(0),
                                                               lane);
          }
          operands.emplace_back(applyCreation(node, vectorIndex, vectorOp));
        }
      } else {

        size_t initialOperandIndex = vectorIndex % node->getOperands().size();

        size_t operandNodeIndex = initialOperandIndex;
        SLPNode* operandNode = (i == 0) ? node : node->getOperand(operandNodeIndex);
        bool useNodeInputs = false;

        size_t nextVectorIndex = vectorsDone[operandNode];
        while (nextVectorIndex >= operandNode->numVectors()) {
          operandNode = node->getOperand(operandNodeIndex % node->getOperands().size());
          nextVectorIndex = vectorsDone[operandNode];
          // Check if we have looped through all operands already.
          if (operandNodeIndex++ == initialOperandIndex + node->numOperands()) {
            useNodeInputs = true;
            break;
          }
        }

        if (useNodeInputs) {
          auto nodeInput = node->getNodeInput(nodeInputsDone[node]++);
          nodeInput.dump();
          if (nodeInput.getType().isa<MemRefType>()) {
            nodeInput = extractMemRefOperand(vectorizableOps[0], builder);
            nodeInput.dump();
          }
          builder.setInsertionPointAfterValue(nodeInput);
          Operation* vectorOp = builder.create<vector::BroadcastOp>(loc, vectorType, nodeInput);
          for (size_t lane = 1; lane < node->numLanes(); ++lane) {
            nodeInput = node->getNodeInput(nodeInputsDone[node]++);
            if (nodeInput.getType().isa<MemRefType>()) {
              nodeInput = extractMemRefOperand(vectorizableOps[0], builder);
              nodeInput.dump();
            }
            builder.setInsertionPointAfterValue(nodeInput);
            vectorOp = builder.create<vector::InsertElementOp>(vectorOp->getLoc(),
                                                               nodeInput,
                                                               vectorOp->getResult(0),
                                                               lane);
          }
          operands.emplace_back(applyCreation(node, vectorIndex, vectorOp));
        } else {
          operands.emplace_back(transform(operandNode, nextVectorIndex));
        }

      }
    }

    auto vectorOp = Operation::create(loc,
                                      firstOp->getName(),
                                      vectorType,
                                      operands,
                                      firstOp->getAttrs(),
                                      firstOp->getSuccessors(),
                                      firstOp->getNumRegions());
    builder.setInsertionPoint(firstOp);
    builder.insert(vectorOp);
    return applyCreation(node, vectorIndex, vectorOp);
  }

  Operation* vectorOp = builder.create<vector::BroadcastOp>(loc, vectorType, firstOp->getResult(0));
  for (size_t lane = 1; lane < node->numLanes(); ++lane) {
    vectorOp = builder.create<vector::InsertElementOp>(vectorOp->getLoc(),
                                                       vectorizableOps[lane]->getResult(0),
                                                       vectorOp->getResult(0),
                                                       lane);
  }

  return applyCreation(node, vectorIndex, vectorOp);
}

Value SLPFunctionTransformer::applyCreation(SLPNode* node,
                                            size_t vectorIndex,
                                            Operation* createdVectorOp) {
  for (size_t lane = 0; lane < node->numLanes(); ++lane) {
    auto* operation = node->getOperation(lane, vectorIndex);
    for (auto* use : operation->getUsers()) {
      for (size_t i = 0; i < use->getNumOperands(); ++i) {
        if (use->getOperand(i) == operation->getResult(0) && !vectorizedOps.count(use)) {
          operandExtractions[use][i] = std::make_pair(createdVectorOp, lane);
          break;
        }
      }
    }
    operandExtractions.erase(operation);
  }
  return createdVectorOp->getResult(0);
}

