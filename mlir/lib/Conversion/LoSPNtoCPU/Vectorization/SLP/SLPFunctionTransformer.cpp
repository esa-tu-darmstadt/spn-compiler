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

SLPFunctionTransformer::SLPFunctionTransformer(std::unique_ptr<SLPNode>&& graph, FuncOp function) : root{
    std::move(graph)}, function{function}, builder{function->getContext()}, typeConverter{
    static_cast<unsigned int>(root->numLanes())} {

}

void SLPFunctionTransformer::transform() {
  builder.setInsertionPoint(root->getOperation(0, 0));
  transform(root.get(), 0);
}

Value SLPFunctionTransformer::transform(SLPNode* node, size_t vectorIndex) {

  vectorsDone[node]++;

  auto vectorType = typeConverter.convertType(node->getOperation(0, 0)->getResultTypes().front());

  auto const& vectorOps = node->getVector(vectorIndex);
  vectorizedOps.insert(vectorOps.begin(), vectorOps.end());
  auto const& firstOp = vectorOps.front();
  auto const& loc = firstOp->getLoc();

  if (areBroadcastable(vectorOps)) {
    auto value = firstOp->getResult(0);
    auto const& insertionPoint = builder.saveInsertionPoint();
    builder.setInsertionPointAfter(firstOp);
    auto broadcast = builder.create<vector::BroadcastOp>(loc, vectorType, value);
    builder.restoreInsertionPoint(insertionPoint);
    return applyCreation(node, vectorIndex, broadcast);
  }

  if (areConsecutiveLoads(vectorOps)) {
    auto batchRead = static_cast<SPNBatchRead>(firstOp);
    auto const& base = batchRead.batchMem();
    auto const& batchIndex = batchRead.batchIndex();
    auto const& offset = getOrCreateConstant(batchRead.sampleIndex());

    auto indices = ValueRange{batchIndex, offset};

    return applyCreation(node, vectorIndex, builder.create<vector::LoadOp>(loc, vectorType, base, indices));
  }

  if (canBeGathered(vectorOps)) {
    SmallVector<Value, 4> elements;
    for (auto const& op : vectorOps) {
      elements.emplace_back(extractMemRefOperand(op));
    }
    return applyCreation(node, vectorIndex, broadcastFirstInsertRest(firstOp, vectorType, elements));
  }

  if (node->isUniform() && firstOp->getNumOperands() > 0) {

    llvm::SmallVector<Value, 2> operands;

    for (size_t i = 0; i < firstOp->getNumOperands(); ++i) {

      if (node->numOperands() == 0) {
        size_t nextIndex = vectorsDone[node];
        if (nextIndex < node->numVectors()) {
          operands.emplace_back(transform(node, nextIndex));
        } else {
          SmallVector<Value, 4> vectorElements;
          for (size_t lane = 0; lane < node->numLanes(); ++lane) {
            auto nodeInput = node->getNodeInput(nodeInputsDone[node]++);
            if (nodeInput.getType().isa<MemRefType>()) {
              nodeInput = extractMemRefOperand(vectorOps[lane]);
            }
            vectorElements.emplace_back(nodeInput);
          }
          auto* vectorOp = broadcastFirstInsertRest(firstOp, vectorType, vectorElements);
          operands.emplace_back(applyCreation(node, vectorIndex, vectorOp));
        }
      } else {

        size_t initialOperandIndex = vectorIndex % node->numOperands();

        size_t operandNodeIndex = initialOperandIndex;
        SLPNode* operandNode = (i == 0) ? node : node->getOperand(operandNodeIndex);
        bool useNodeInputs = false;

        size_t nextVectorIndex = vectorsDone[operandNode];
        while (nextVectorIndex >= operandNode->numVectors()) {
          operandNode = node->getOperand(operandNodeIndex % node->numOperands());
          nextVectorIndex = vectorsDone[operandNode];
          // Check if we have looped through all operands already.
          if (operandNodeIndex++ == initialOperandIndex + node->numOperands()) {
            useNodeInputs = true;
            break;
          }
        }

        if (useNodeInputs) {
          SmallVector<Value, 4> vectorElements;
          for (size_t lane = 0; lane < node->numLanes(); ++lane) {
            auto nodeInput = node->getNodeInput(nodeInputsDone[node]++);
            if (nodeInput.getType().isa<MemRefType>()) {
              nodeInput = extractMemRefOperand(vectorOps[lane]);
            }
            vectorElements.emplace_back(nodeInput);
          }
          auto* vectorOp = broadcastFirstInsertRest(firstOp, vectorType, vectorElements);
          operands.emplace_back(applyCreation(node, vectorIndex, vectorOp));
        } else {
          operands.emplace_back(transform(operandNode, nextVectorIndex));
        }

      }
    }

    Operation* vectorOp;

    if (dyn_cast<SPNMul>(firstOp)) {
      vectorOp = builder.create<MulFOp>(loc, operands);
    } else if (dyn_cast<SPNAdd>(firstOp)) {
      vectorOp = builder.create<AddFOp>(loc, operands);
    } else {
      vectorOp = builder.insert(Operation::create(loc,
                                                  firstOp->getName(),
                                                  vectorType,
                                                  operands,
                                                  firstOp->getAttrs(),
                                                  firstOp->getSuccessors(),
                                                  firstOp->getNumRegions()));
    }
    return applyCreation(node, vectorIndex, vectorOp);
  }

  llvm::SmallVector<Value, 4> vectorElements;
  for (auto op : vectorOps) {
    vectorElements.emplace_back(op->getResult(0));
  }
  return applyCreation(node, vectorIndex, broadcastFirstInsertRest(firstOp, vectorType, vectorElements));
}

Value SLPFunctionTransformer::applyCreation(SLPNode* node, size_t vectorIndex, Operation* createdVectorOp) {

  bool isBroadcast = dyn_cast<vector::BroadcastOp>(createdVectorOp);

  for (size_t lane = 0; lane < node->numLanes(); ++lane) {
    auto* operation = node->getOperation(lane, vectorIndex);
    if (isBroadcast && operation == node->getOperation(0, vectorIndex)) {
      continue;
    }
    bool createExtraction = false;
    for (auto* use : operation->getUsers()) {
      if (vectorizedOps.find(use) == std::end(vectorizedOps)) {
        createExtraction = true;
        break;
      }
    }

    if (createExtraction) {

      auto const& source = createdVectorOp->getResult(0);
      auto const& position = getOrCreateConstant(lane, false);

      auto const& insertionPoint = builder.saveInsertionPoint();
      builder.setInsertionPointAfter(createdVectorOp);
      auto element = builder.create<vector::ExtractElementOp>(builder.getUnknownLoc(), source, position);
      builder.restoreInsertionPoint(insertionPoint);
      operation->replaceAllUsesWith(element);

    }
    operation->dropAllUses();
    operation->remove();
  }
  return createdVectorOp->getResult(0);
}

Operation* SLPFunctionTransformer::broadcastFirstInsertRest(Operation* beforeOp,
                                                            Type const& vectorType,
                                                            SmallVector<Value, 4>& elements) {
  Operation* vectorOp = builder.create<vector::BroadcastOp>(beforeOp->getLoc(), vectorType, elements[0]);
  for (size_t i = 1; i < elements.size(); ++i) {
    auto const& position = getOrCreateConstant(i, false);
    vectorOp =
        builder.create<vector::InsertElementOp>(beforeOp->getLoc(), elements[i], vectorOp->getResult(0), position);
  }
  return vectorOp;
}

Value SLPFunctionTransformer::getOrCreateConstant(unsigned index, bool asIndex) {
  auto& createdConstants = asIndex ? createdIndexConstants : createdUnsignedConstants;
  if (!createdConstants.count(index)) {
    auto const& insertionPoint = builder.saveInsertionPoint();
    auto const& attribute = asIndex ? builder.getIndexAttr(index) : builder.getIntegerAttr(builder.getI32Type(), index);
    builder.setInsertionPointToStart(&function.body().front());
    createdConstants[index] = builder.create<ConstantOp>(builder.getUnknownLoc(), attribute);
    builder.restoreInsertionPoint(insertionPoint);
  }
  return createdConstants[index];
}

Value SLPFunctionTransformer::extractMemRefOperand(Operation* op) {
  Value base;
  ValueRange indices;
  if (auto batchRead = dyn_cast<SPNBatchRead>(op)) {
    base = batchRead.batchMem();
    auto const& batchIndex = batchRead.batchIndex();
    auto const& sampleIndex = batchRead.sampleIndex();
    auto const& memRefTuple = std::make_tuple(base, batchIndex, sampleIndex);

    if (memRefLoads.count(memRefTuple)) {
      return memRefLoads[memRefTuple];
    }

    indices = {batchIndex, getOrCreateConstant(sampleIndex)};

  } else {
    assert(false);
  }

  auto const& insertionPoint = builder.saveInsertionPoint();
  builder.setInsertionPoint(op);
  auto load = builder.create<LoadOp>(op->getLoc(), base, indices);
  builder.restoreInsertionPoint(insertionPoint);
  return load;
}