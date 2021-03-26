//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPN/LoSPNDialect.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPVectorizationPass.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPUtil.h"
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
  SLPGraph graph(seeds.front(), 3);
  transform(graph);

}

void SLPVectorizationPass::transform(SLPGraph const& graph) {
  //graph.dump();
  std::map<SLPNode const*, unsigned> vectorsDone;
  auto rootOperation = transform(graph.getRoot(), 0, vectorsDone);
  llvm::dbgs() << extractions.size() << "\n";
  assert(false);
  // Update users of vector operations that aren't contained in the graph.
}

mlir::Value SLPVectorizationPass::transform(SLPNode const& node,
                                            size_t vectorIndex,
                                            std::map<SLPNode const*, unsigned>& vectorsDone) {

  node.dump();
  llvm::dbgs() << "\n";
  if (std::uintptr_t(node.getOperation(0, 0)) == 0x716008) {
    llvm::dbgs() << "\n";
  }

  vectorsDone[&node]++;

  LoSPNVectorizationTypeConverter typeConverter{static_cast<unsigned int>(node.numLanes())};
  auto vectorType = typeConverter.convertType(node.getResultType());
  auto builder = OpBuilder{&getContext()};

  auto const& vectorizableOps = node.getVector(vectorIndex);
  auto const& firstOp = vectorizableOps.front();
  builder.setInsertionPointAfter(firstOp);

  if (isBroadcastable(vectorizableOps)) {
    auto value = firstOp->getResult(0);
    auto vectorOp = builder.create<vector::BroadcastOp>(firstOp->getLoc(), vectorType, value);
    return updateExtractions(node, vectorIndex, vectorOp)->getResult(0);
  }

  if (areConsecutiveLoads(vectorizableOps)) {
    auto base = firstOp->getResult(0);
    auto zero = builder.create<ConstantOp>(firstOp->getLoc(), builder.getIndexAttr(0));
    auto indices = ValueRange{zero.getResult()};
    auto vectorOp = builder.create<vector::LoadOp>(firstOp->getLoc(), vectorType, base, indices);
    return updateExtractions(node, vectorIndex, vectorOp)->getResult(0);
  }

  if (node.isUniform()) {

    llvm::SmallVector<Value, 2> operands;

    for (size_t i = 0; i < firstOp->getNumOperands(); ++i) {
      size_t operandNodeIndex = vectorIndex % node.getOperands().size();
      auto* operandNode = (i == 0) ? &node : &node.getOperand(operandNodeIndex);
      size_t nextIndex = vectorsDone[operandNode];
      while (nextIndex >= operandNode->numVectors()) {
        operandNode = &node.getOperand(operandNodeIndex++ % node.getOperands().size());
        nextIndex = vectorsDone[operandNode];
      }
      operands.emplace_back(transform(*operandNode, nextIndex, vectorsDone));
    }

    auto vectorOp = Operation::create(firstOp->getLoc(),
                                      firstOp->getName(),
                                      vectorType,
                                      operands,
                                      firstOp->getAttrs(),
                                      firstOp->getSuccessors(),
                                      firstOp->getNumRegions());
    builder.insert(vectorOp);
    return updateExtractions(node, vectorIndex, vectorOp)->getResult(0);
  }

  Operation* vectorOp = builder.create<vector::BroadcastOp>(firstOp->getLoc(), vectorType, firstOp->getResult(0));
  for (size_t lane = 1; lane < node.numLanes(); ++lane) {
    vectorOp = builder.create<vector::InsertElementOp>(vectorOp->getLoc(),
                                                       vectorizableOps[lane]->getResult(0),
                                                       vectorOp->getResult(0),
                                                       lane);
  }

  return updateExtractions(node, vectorIndex, vectorOp)->getResult(0);
}

mlir::Operation* SLPVectorizationPass::updateExtractions(SLPNode const& node,
                                                         size_t const& vectorIndex,
                                                         Operation* vectorOp) {
  for (size_t lane = 0; lane < node.numLanes(); ++lane) {
    auto* operation = node.getOperation(lane, vectorIndex);
    for (auto* use : operation->getUsers()) {
      for (size_t i = 0; i < use->getNumOperands(); ++i) {
        if (use->getOperand(i) == operation->getResult(0)) {
          extractions[use][i] = std::make_pair(vectorOp, i);
          break;
        }
      }
    }
    extractions.erase(operation);
  }
  return vectorOp;
}

std::unique_ptr<mlir::Pass> mlir::spn::createSLPVectorizationPass() {
  return std::make_unique<SLPVectorizationPass>();
}
