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
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Rewrite/PatternApplicator.h"
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
      for (auto& vector : node->getVectors()) {
        for (auto* op : vector) {
          parentMapping[op] = node;
        }
      }
      for (auto const& operand : node->getOperands()) {
        worklist.emplace(operand);
      }
    }
    return parentMapping;
  }

  llvm::SmallVector<SLPNode*> postOrder(SLPNode* root) {
    llvm::SmallVector<SLPNode*> order;
    for (auto* operand : root->getOperands()) {
      order.append(postOrder(operand));
    }
    order.emplace_back(root);
    return order;
  }

  /// A custom PatternRewriter that does not fold operations.
  class NoFoldPatternRewriter : public mlir::PatternRewriter {
  public:
    explicit NoFoldPatternRewriter(mlir::MLIRContext* ctx) : PatternRewriter(ctx) {}
  };

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
  // printSeedTree(seeds.front());
  SLPGraphBuilder builder{3};
  auto graph = builder.build(seeds.front());
  // graph->dumpGraph();
  // ==== //

  OwningRewritePatternList patterns;
  auto const& parentMapping = computeParentNodesMapping(graph.get());
  llvm::DenseMap<SLPNode*, llvm::SmallVector<Operation*>> vectorsByNode;
  populateSLPVectorizationPatterns(patterns, &getContext(), parentMapping, vectorsByNode);
  FrozenRewritePatternList frozenPatterns(std::move(patterns));

  // Traverse the SLP graph in postorder and apply the vectorization patterns.
  NoFoldPatternRewriter rewriter(&getContext());
  PatternApplicator applicator(frozenPatterns);
  applicator.applyDefaultCostModel();

  // Marks operations that can be deleted.
  // We delete them *after* SLP graph conversion to avoid running into NULL operands during conversion.
  llvm::SmallPtrSet<Operation*, 32> erasableOps;
  for (auto* node : postOrder(graph.get())) {

    // Stores escaping uses for vector extractions that might be necessary later on.
    llvm::DenseMap<Operation*, llvm::SmallPtrSet<Operation*, 8>> escapingUses;

    // Also traverse nodes in postorder to properly handle multinodes.
    auto it = node->getVectors().rbegin();
    while (it != node->getVectors().rend()) {

      // Gather escaping uses *now* because the conversion process adds more that *do not* need extractions.
      for (size_t lane = 0; lane < it->size(); ++lane) {
        auto* vectorOp = (*it)[lane];
        for (auto* user : vectorOp->getUsers()) {
          if (!parentMapping.count(user)) {
            escapingUses[vectorOp].insert(user);
          }
        }
      }

      // Rewrite vector by applying any matching pattern.
      vectorsByNode[node].resize(node->numVectors());
      auto result = applicator.matchAndRewrite(it->front(), rewriter);
      if (result.failed()) {
        it->front()->emitOpError("SLP pattern application failed (did you forget to specify the pattern?)");
      }
      ++it;
    }

    // Gather operations that can be erased and create vector extractions using those that need to stay.
    for (size_t vectorIndex = node->numVectors(); vectorIndex-- > 0;) {
      auto const& vector = node->getVector(vectorIndex);
      for (size_t lane = 0; lane < vector.size(); ++lane) {
        auto* vectorOp = vector[lane];
        bool erasable = true;
        for (auto& use : vectorOp->getUses()) {
          auto* user = use.getOwner();
          if (!parentMapping.count(user)) {
            erasable = false;
            // Filters out operations that were created during the conversion process.
            if (!escapingUses[vectorOp].contains(user)) {
              continue;
            }
            auto const& source = vectorsByNode[node][vectorIndex]->getResult(0);
            rewriter.setInsertionPoint(user);
            auto extractOp = rewriter.create<vector::ExtractElementOp>(vectorOp->getLoc(), source, lane);
            auto const& oldOperand = user->getOperand(use.getOperandNumber());
            user->setOperand(use.getOperandNumber(), extractOp);
            if (oldOperand.getUses().empty() && !oldOperand.isa<BlockArgument>()) {
              erasableOps.insert(oldOperand.getDefiningOp());
            }
          }
        }
        if (erasable) {
          erasableOps.insert(vectorOp);
        }
      }
    }
  }

  for (auto* op : erasableOps) {
    op->dropAllUses();
    rewriter.eraseOp(op);
  }

  getOperation()->dump();

  // If an SLP node failed to vectorize completely, fail the pass.
  for (auto const& entry: vectorsByNode) {
    if (entry.first->numVectors() != entry.second.size()) {
      llvm::dbgs() << "Failed to vectorize node:\n";
      entry.first->dump();
      signalPassFailure();
    }
  }

}

std::unique_ptr<mlir::Pass> mlir::spn::createSLPVectorizationPass() {
  return std::make_unique<SLPVectorizationPass>();
}
