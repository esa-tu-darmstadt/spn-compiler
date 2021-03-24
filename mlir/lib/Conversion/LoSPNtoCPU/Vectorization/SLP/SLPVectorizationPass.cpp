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

  SLPGraph graph(seeds.front(), 3);
  transform(graph);

}

void SLPVectorizationPass::transform(SLPGraph const& graph) {
  //graph.dump();
  auto* rootOperation = transform(graph.getRoot(), 0, 1);
  assert(false);
  // Update users of vector operations that aren't contained in the graph.
}

mlir::Operation* SLPVectorizationPass::transform(SLPNode const& node, size_t vectorIndex, size_t spill) {

  node.dump();
  llvm::dbgs() << "\n";
  if (std::uintptr_t(node.getOperation(0, 0)) == 0x9033b8) {
    llvm::dbgs() << "\n";
  }

  LoSPNVectorizationTypeConverter typeConverter{static_cast<unsigned int>(node.numLanes())};
  auto vectorType = typeConverter.convertType(node.getResultType());
  auto builder = OpBuilder{&getContext()};

  auto const& vectorizableOps = node.getVector(vectorIndex);
  auto const& firstOp = vectorizableOps.front();
  builder.setInsertionPointAfter(firstOp);

  Operation* vectorOp;

  if (isBroadcastable(vectorizableOps)) {
    auto value = firstOp->getResult(0);
    vectorOp = builder.create<vector::BroadcastOp>(firstOp->getLoc(), vectorType, value);
  } else if (node.isUniform()) {
    if (areConsecutiveLoads(vectorizableOps)) {
      auto base = firstOp->getResult(0);
      auto zero = builder.create<ConstantOp>(firstOp->getLoc(), builder.getIndexAttr(0));
      auto indices = ValueRange{zero.getResult()};
      vectorOp = builder.create<vector::LoadOp>(firstOp->getLoc(), vectorType, base, indices);
    } else {
      llvm::SmallVector<Value, 2> operands;
      for (size_t i = 0; i < firstOp->getNumOperands(); ++i) {

        SLPNode const* operandNode = &node;
        size_t operandNodeIndex = 0;

        unsigned availableOperands = node.numVectors();
        unsigned usedOperands = vectorIndex * firstOp->getNumOperands() + spill;
        size_t nextIndex = usedOperands + i;
        size_t nodeSpill = spill;

        while (availableOperands <= nextIndex) {
          if (operandNodeIndex == 0) {
            nodeSpill = ((firstOp->getNumOperands() - 1) * node.numVectors()) + spill;
          } else {
            nodeSpill -= availableOperands;
          }
          operandNode = &node.getOperand(operandNodeIndex++);
          nextIndex -= availableOperands;
          availableOperands = operandNode->numVectors();
        }

        operands.emplace_back(transform(*operandNode, nextIndex, nodeSpill)->getResult(0));
      }
      vectorOp = Operation::create(firstOp->getLoc(),
                                   firstOp->getName(),
                                   vectorType,
                                   operands,
                                   firstOp->getAttrs(),
                                   firstOp->getSuccessors(),
                                   firstOp->getNumRegions());
      builder.insert(vectorOp);
    }

  } else {
    vectorOp = builder.create<vector::BroadcastOp>(firstOp->getLoc(), vectorType, firstOp->getResult(0));
    for (size_t lane = 1; lane < node.numLanes(); ++lane) {
      vectorOp = builder.create<vector::InsertElementOp>(vectorOp->getLoc(),
                                                         vectorizableOps[lane]->getResult(0),
                                                         vectorOp->getResult(0),
                                                         lane);
    }
  }

  vectorOp->dump();
  updateExtractions(node, vectorIndex, vectorOp);

  return vectorOp;
}

void SLPVectorizationPass::updateExtractions(SLPNode const& node, size_t const& vectorIndex, Operation* vectorOp) {
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
}

std::unique_ptr<mlir::Pass> mlir::spn::createSLPVectorizationPass() {
  return std::make_unique<SLPVectorizationPass>();
}

