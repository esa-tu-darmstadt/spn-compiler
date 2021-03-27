//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPN/LoSPNDialect.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPVectorizationPass.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPUtil.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPGraphBuilder.h"
#include "LoSPNtoCPU/Vectorization/TargetInformation.h"
#include "LoSPNtoCPU/LoSPNtoCPUTypeConverter.h"
#include "LoSPNtoCPU/Vectorization/LoSPNVectorizationTypeConverter.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir::spn::low::slp;

// Anonymous namespace for helper functions.
namespace {

  bool isBroadcastable(std::vector<mlir::Operation*> const& vector) {
    return std::all_of(std::begin(vector), std::end(vector), [&](mlir::Operation* op) {
      return op == vector.front();
    });
  }

  /// For debugging purposes.
  void printOperation(mlir::Operation* op) {
    llvm::dbgs() << "node_" << op << "[label=\"" << op->getName().getStringRef() << "\\n" << op;
    if (auto constantOp = llvm::dyn_cast<mlir::ConstantOp>(op)) {
      if (constantOp.value().getType().isIntOrIndex()) {
        llvm::dbgs() << "\\nvalue: " << std::to_string(constantOp.value().dyn_cast<mlir::IntegerAttr>().getInt());
      } else if (constantOp.value().getType().isIntOrFloat()) {
        llvm::dbgs() << "\\nvalue: "
                     << std::to_string(constantOp.value().dyn_cast<mlir::FloatAttr>().getValueAsDouble());
      }
    } else if (auto batchReadOp = llvm::dyn_cast<mlir::spn::low::SPNBatchRead>(op)) {
      llvm::dbgs() << "\\nbatch mem: " << batchReadOp.batchMem().dyn_cast<mlir::BlockArgument>().getArgNumber();
      llvm::dbgs() << "\\nbatch index: " << batchReadOp.batchMem().dyn_cast<mlir::BlockArgument>().getArgNumber();
      llvm::dbgs() << "\\nsample index: " << batchReadOp.sampleIndex();
    }
    llvm::dbgs() << "\", fillcolor=\"#a0522d\"];\n";
  }

  /// For debugging purposes.
  void printEdge(mlir::Operation* src, mlir::Operation* dst, size_t index) {
    llvm::dbgs() << "node_" << src << " -> node_" << dst << "[label=\"" << std::to_string(index) << "\"];\n";
  }

  /// For debugging purposes.
  void printSeedTree(std::vector<mlir::Operation*> const& operations) {
    std::vector<mlir::Operation*> nodes;
    std::vector<std::tuple<mlir::Operation*, mlir::Operation*, size_t>> edges;

    std::stack<mlir::Operation*> worklist;
    for (auto& op : operations) {
      worklist.push(op);
    }

    while (!worklist.empty()) {

      auto op = worklist.top();
      worklist.pop();

      if (std::find(std::begin(nodes), std::end(nodes), op) == nodes.end()) {
        nodes.emplace_back(op);
        for (size_t i = 0; i < op->getNumOperands(); ++i) {
          auto const& operand = op->getOperand(i);
          if (operand.getDefiningOp() != nullptr) {
            edges.emplace_back(std::make_tuple(op, operand.getDefiningOp(), i));
            worklist.push(operand.getDefiningOp());
          }
        }
      }
    }

    llvm::dbgs() << "digraph debug_graph {\n";
    llvm::dbgs() << "rankdir = BT;\n";
    llvm::dbgs() << "node[shape=box];\n";
    for (auto& op : nodes) {
      printOperation(op);
    }
    for (auto& edge : edges) {
      printEdge(std::get<0>(edge), std::get<1>(edge), std::get<2>(edge));
    }
    llvm::dbgs() << "}\n";
  }

}

void SLPVectorizationPass::runOnOperation() {
  llvm::StringRef funcName = getOperation().getName();
  if (!funcName.contains("task_")) {
    return;
  }

  auto function = getOperation();

  auto& seedAnalysis = getAnalysis<SeedAnalysis>();

  auto computationType = function.getArguments().back().getType().dyn_cast<MemRefType>().getElementType();
  auto width = TargetInformation::nativeCPUTarget().getHWVectorEntries(computationType);

  auto seeds = seedAnalysis.getSeeds(width, seedAnalysis.getOpDepths(), SearchMode::UseBeforeDef);
  assert(!seeds.empty() && "couldn't find a seed!");
  printSeedTree(seeds.front());
  SLPGraphBuilder builder{3};
  auto graph = builder.build(seeds.front());
  transform(graph.get());

}

void SLPVectorizationPass::transform(SLPNode* root) {
  root->dumpGraph();
  std::map<SLPNode*, size_t> vectorsDone;
  std::map<SLPNode*, size_t> nodeInputsDone;
  auto rootOperation = transform(root, 0, vectorsDone, nodeInputsDone);
  getOperation()->dump();
  llvm::dbgs() << extractions.size() << "\n";
  assert(false);
  // Update users of vector operations that aren't contained in the graph.
}

mlir::Value SLPVectorizationPass::transform(SLPNode* node,
                                            size_t vectorIndex,
                                            std::map<SLPNode*, size_t>& vectorsDone,
                                            std::map<SLPNode*, size_t>& nodeInputsDone) {

  node->dump();
  llvm::dbgs() << "\n";
  if (std::uintptr_t(node->getOperation(0, 0)) == 0x716008) {
    llvm::dbgs() << "\n";
  }

  vectorsDone[node]++;

  LoSPNVectorizationTypeConverter typeConverter{static_cast<unsigned int>(node->numLanes())};
  auto vectorType = typeConverter.convertType(node->getResultType());
  auto builder = OpBuilder{&getContext()};

  auto const& vectorizableOps = node->getVector(vectorIndex);
  auto const& firstOp = vectorizableOps.front();
  auto const& loc = firstOp->getLoc();
  builder.setInsertionPoint(firstOp);

  if (isBroadcastable(vectorizableOps)) {
    auto value = firstOp->getResult(0);
    auto vectorOp = builder.create<vector::BroadcastOp>(loc, vectorType, value);
    return applyCreation(node, vectorIndex, vectorOp);
  }

  if (areConsecutiveLoads(vectorizableOps)) {
    auto base = firstOp->getResult(0);
    auto zero = builder.create<ConstantOp>(loc, builder.getIndexAttr(0));
    auto indices = ValueRange{zero.getResult()};
    auto vectorOp = builder.create<vector::LoadOp>(loc, vectorType, base, indices);
    return applyCreation(node, vectorIndex, vectorOp);
  }

  if (node->isUniform() && firstOp->getNumOperands() > 0) {

    llvm::SmallVector<Value, 2> operands;

    for (size_t i = 0; i < firstOp->getNumOperands(); ++i) {

      if (node->numOperands() == 0) {
        size_t nextIndex = vectorsDone[node];
        if (nextIndex < node->numVectors()) {
          operands.emplace_back(transform(node, nextIndex, vectorsDone, nodeInputsDone));
        } else {
          auto nodeInput = node->getNodeInput(nodeInputsDone[node]++);
          nodeInput.dump();
          Operation* vectorOp = builder.create<vector::BroadcastOp>(loc, vectorType, nodeInput);
          for (size_t lane = 1; lane < node->numLanes(); ++lane) {
            nodeInput = node->getNodeInput(nodeInputsDone[node]++);
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
          Operation* vectorOp = builder.create<vector::BroadcastOp>(loc, vectorType, nodeInput);
          for (size_t lane = 1; lane < node->numLanes(); ++lane) {
            nodeInput = node->getNodeInput(nodeInputsDone[node]++);
            vectorOp = builder.create<vector::InsertElementOp>(vectorOp->getLoc(),
                                                               nodeInput,
                                                               vectorOp->getResult(0),
                                                               lane);
          }
          operands.emplace_back(applyCreation(node, vectorIndex, vectorOp));
        } else {
          operands.emplace_back(transform(operandNode, nextVectorIndex, vectorsDone, nodeInputsDone));
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

mlir::Value SLPVectorizationPass::applyCreation(SLPNode* node,
                                                size_t vectorIndex,
                                                Operation* createdVectorOp) {
  for (size_t lane = 0; lane < node->numLanes(); ++lane) {
    auto* operation = node->getOperation(lane, vectorIndex);
    for (auto* use : operation->getUsers()) {
      for (size_t i = 0; i < use->getNumOperands(); ++i) {
        if (use->getOperand(i) == operation->getResult(0)) {
          extractions[use][i] = std::make_pair(createdVectorOp, i);
          break;
        }
      }
    }
    extractions.erase(operation);
  }
  return createdVectorOp->getResult(0);
}

std::unique_ptr<mlir::Pass> mlir::spn::createSLPVectorizationPass() {
  return std::make_unique<SLPVectorizationPass>();
}
