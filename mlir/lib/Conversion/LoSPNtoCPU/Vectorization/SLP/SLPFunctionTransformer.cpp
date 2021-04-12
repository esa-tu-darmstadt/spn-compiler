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

  // Create extractions for operations that need values from the SLPGraph, but are located outside of it.
  for (auto const& entry : operandExtractions) {
    auto* op = entry.getFirst();
    for (auto const& extraction : entry.getSecond()) {

      auto const& vector = extraction.second.first->getResult(0);
      auto const& position = getOrCreateConstant(extraction.second.second);

      builder.setInsertionPoint(op);
      auto element = builder.create<vector::ExtractElementOp>(op->getLoc(), vector, position);
      op->setOperand(extraction.first, element);

    }
  }
}

Value SLPFunctionTransformer::transform(SLPNode* node, size_t vectorIndex) {

  vectorsDone[node]++;

  auto vectorType = typeConverter.convertType(node->getResultType());

  auto const& vectorizableOps = node->getVector(vectorIndex);
  vectorizedOps.insert(vectorizableOps.begin(), vectorizableOps.end());
  auto const& firstOp = vectorizableOps.front();
  auto const& loc = firstOp->getLoc();

  if (areBroadcastable(vectorizableOps)) {
    auto value = firstOp->getResult(0);
    return applyCreation(node, vectorIndex, builder.create<vector::BroadcastOp>(loc, vectorType, value), true);
  }

  if (areConsecutiveLoads(vectorizableOps)) {
    auto batchRead = static_cast<SPNBatchRead>(firstOp);
    auto const& base = batchRead.batchMem();
    auto const& batchIndex = batchRead.batchIndex();
    auto const& offset = getOrCreateConstant(batchRead.sampleIndex());

    auto indices = ValueRange{batchIndex, offset};

    return applyCreation(node, vectorIndex, builder.create<vector::LoadOp>(loc, vectorType, base, indices));
  }

  if (canBeGathered(vectorizableOps)) {
    SmallVector<Value, 4> elements;
    for (auto const& op : vectorizableOps) {
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
              nodeInput = extractMemRefOperand(vectorizableOps[lane]);
            }
            vectorElements.emplace_back(nodeInput);
          }
          auto* vectorOp = broadcastFirstInsertRest(firstOp, vectorType, vectorElements);
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
          SmallVector<Value, 4> vectorElements;
          for (size_t lane = 0; lane < node->numLanes(); ++lane) {
            auto nodeInput = node->getNodeInput(nodeInputsDone[node]++);
            if (nodeInput.getType().isa<MemRefType>()) {
              nodeInput = extractMemRefOperand(vectorizableOps[lane]);
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

    auto vectorOp = Operation::create(loc,
                                      firstOp->getName(),
                                      vectorType,
                                      operands,
                                      firstOp->getAttrs(),
                                      firstOp->getSuccessors(),
                                      firstOp->getNumRegions());
    return applyCreation(node, vectorIndex, builder.insert(vectorOp));
  }

  llvm::SmallVector<Value, 4> vectorElements;
  for (auto op : vectorizableOps) {
    vectorElements.emplace_back(op->getResult(0));
  }
  return applyCreation(node, vectorIndex, broadcastFirstInsertRest(firstOp, vectorType, vectorElements));
}

Value SLPFunctionTransformer::applyCreation(SLPNode* node,
                                            size_t vectorIndex,
                                            Operation* createdVectorOp,
                                            bool keepFirst) {
  for (size_t lane = 0; lane < node->numLanes(); ++lane) {
    auto* operation = node->getOperation(lane, vectorIndex);
    for (auto* use : operation->getUsers()) {
      // Happens when creating broadcast operations.
      if (use == createdVectorOp) {
        continue;
      }
      for (size_t i = 0; i < use->getNumOperands(); ++i) {
        if (use->getOperand(i) == operation->getResult(0) && !vectorizedOps.count(use)) {
          operandExtractions[use][i] = std::make_pair(createdVectorOp, lane);
          break;
        }
      }
    }
    operandExtractions.erase(operation);
    if (lane == 0 && keepFirst) {
      continue;
    }
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