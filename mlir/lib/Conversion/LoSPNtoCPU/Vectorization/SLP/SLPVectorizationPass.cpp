//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPN/LoSPNDialect.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPVectorizationPass.h"
#include "LoSPNtoCPU/Vectorization/TargetInformation.h"
#include "LoSPNtoCPU/LoSPNtoCPUTypeConverter.h"
#include "LoSPNtoCPU/Vectorization/LoSPNVectorizationTypeConverter.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

// Anonymous namespace for helper functions.
namespace {

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

void mlir::spn::low::slp::SLPVectorizationPass::runOnOperation() {
  llvm::StringRef funcName = getOperation().getName();
  if (!funcName.contains("task_")) {
    return;
  }

  ConversionTarget target(getContext());

  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalDialect<mlir::scf::SCFDialect>();
  target.addLegalDialect<mlir::vector::VectorDialect>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
  target.addLegalOp<FuncOp>();

  target.addLegalDialect<mlir::spn::low::LoSPNDialect>();

  auto function = getOperation();

  auto& seedAnalysis = getAnalysis<SeedAnalysis>();

  auto computationType = function.getArguments().back().getType().dyn_cast<MemRefType>().getElementType();
  auto width = TargetInformation::nativeCPUTarget().getHWVectorEntries(computationType);

  auto seeds = seedAnalysis.getSeeds(width, seedAnalysis.getOpDepths(), SearchMode::UseBeforeDef);
  assert(!seeds.empty() && "couldn't find a seed!");

  SLPGraph graph(seeds.front(), 3);
  transform(graph);

}

void mlir::spn::low::slp::SLPVectorizationPass::transform(SLPGraph& graph) {
  graph.dump();
  auto* rootOperation = transform(graph.getRoot(), true);
  assert(false);
  // Update users of vector operations that aren't contained in the graph.
  for (auto const& entry : extractOps) {
    for (auto const& extractOpData : entry.second) {
      extractOpData.first->setOperand(extractOpData.second, nullptr);
    }
  }
}

mlir::Operation* mlir::spn::low::slp::SLPVectorizationPass::transform(SLPNode& node, bool isRoot) {
  LoSPNVectorizationTypeConverter typeConverter{4};
  for (size_t vectorIndex = 0; vectorIndex < node.numVectors(); ++vectorIndex) {
    if (node.isUniform()) {

      for (size_t lane = 0; lane < node.numLanes(); ++lane) {
        auto* operation = node.getOperation(lane, vectorIndex);

        auto type = VectorType::get(node.numLanes(), operation->getResult(0).getType());
        auto operands = llvm::SmallVector<Value, 2>{};
        auto* vectorOperation = Operation::create(operation->getLoc(),
                                                  operation->getName(),
                                                  type,
                                                  operands,
                                                  operation->getAttrs(),
                                                  operation->getSuccessors(),
                                                  operation->getNumRegions());

        operation->dropAllUses();

      }
    } else {

    }
  }
  bool isMixed = false;
  assert(false);
  extractOps[nullptr];
  return node.getVector(0).front();
}

std::unique_ptr<mlir::Pass> mlir::spn::createSLPVectorizationPass() {
  return std::make_unique<mlir::spn::low::slp::SLPVectorizationPass>();
}

