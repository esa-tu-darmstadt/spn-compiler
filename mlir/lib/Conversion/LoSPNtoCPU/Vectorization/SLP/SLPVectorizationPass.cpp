//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPN/LoSPNDialect.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPVectorizationPass.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPUtil.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPGraphBuilder.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPVectorizationPatterns.h"
#include "LoSPNtoCPU/Vectorization/TargetInformation.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <queue>

using namespace mlir::spn::low::slp;

// Anonymous namespace for helper functions.
namespace {

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
    for (auto& edge : edges) {
      llvm::dbgs() << "node_" << std::get<0>(edge) << " -> node_" << std::get<1>(edge) << "[label=\""
                   << std::to_string(std::get<2>(edge)) << "\"];\n";
    }
    llvm::dbgs() << "}\n";
  }

  llvm::DenseMap<mlir::Operation*, SLPNode*> computeParentNodesMapping(SLPNode* root) {
    llvm::DenseMap<mlir::Operation*, SLPNode*> parentMapping;
    std::queue<SLPNode*> worklist;
    worklist.emplace(root);
    while (!worklist.empty()) {
      auto* node = worklist.front();
      worklist.pop();
      for(auto& vector : node->getVectors()) {
        for(auto* op : vector) {
          parentMapping[op] = node;
        }
      }
      for (auto const& operand : node->getOperands()) {
        worklist.emplace(operand);
      }
    }
    return parentMapping;
  }

  /// Define a custom PatternRewriter for use by the driver.
  class MyPatternRewriter : public mlir::PatternRewriter {
  public:
    MyPatternRewriter(mlir::MLIRContext *ctx) : PatternRewriter(ctx) {}

    /// Override the necessary PatternRewriter hooks here.
  };

/// Apply the custom driver to `op`.
  void applyMyPatternDriver(mlir::Operation *op,
                            mlir::FrozenRewritePatternList const& patterns) {
    // Initialize the custom PatternRewriter.
    MyPatternRewriter rewriter(op->getContext());

    // Create the applicator and apply our cost model.
    mlir::PatternApplicator applicator(patterns);
    applicator.applyDefaultCostModel();

    //auto resulttest = patterns.getNativePatterns().begin()->matchAndRewrite(op, rewriter);

    // Try to match and apply a pattern.
    mlir::LogicalResult result = applicator.matchAndRewrite(op, rewriter);
    if (failed(result)) {
      // ... No patterns were applied.
    }
    // ... A pattern was successfully applied.
  }

} // End anonymous namespace.

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
  //printSeedTree(seeds.front());
  SLPGraphBuilder builder{3};
  auto graph = builder.build(seeds.front());
  //SLPFunctionTransformer transformer(std::move(graph), function);
  //transformer.transform();
  // ==== //

  OwningRewritePatternList patterns;
  auto const& parentMapping = computeParentNodesMapping(graph.get());
  populateSLPVectorizationPatterns(patterns, &getContext(), parentMapping);
  FrozenRewritePatternList frozenPatterns(std::move(patterns));
  for(auto const& entry : parentMapping) {
    if(dyn_cast<SPNConstant>(entry.first)) {
      llvm::dbgs() << "constant op: " << entry.first << "\n";
      entry.second->dump();
      //applyMyPatternDriver(entry.first, frozenPatterns);
    }
    entry.first->dump();
    bool erased;
    auto success = applyOpPatternsAndFold(entry.first, frozenPatterns, &erased);

    assert(success.succeeded());

    if (success.failed()) {
      signalPassFailure();
    }
  }

  getOperation()->dump();
}

std::unique_ptr<mlir::Pass> mlir::spn::createSLPVectorizationPass() {
  return std::make_unique<SLPVectorizationPass>();
}
