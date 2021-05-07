//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPU/Vectorization/SLP/Util.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include <queue>

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::low;
using namespace mlir::spn::low::slp;

bool slp::vectorizable(Operation* op) {
  return (op->hasTrait<OpTrait::spn::low::VectorizableOp>() || op->hasTrait<OpTrait::ConstantLike>())
      && op->hasTrait<OpTrait::OneResult>() && op->getResult(0).getType().isIntOrFloat();
}

bool slp::vectorizable(Value const& value) {
  // Block arguments don't have defining ops (they can still be put in a vector by other means).
  if (auto* definingOp = value.getDefiningOp()) {
    if (!vectorizable(definingOp)) {
      return false;
    }
  }
  return value.getType().isIntOrFloat();
}

bool slp::consecutiveLoads(Value const& lhs, Value const& rhs) {
  if (lhs == rhs || lhs.isa<BlockArgument>() || rhs.isa<BlockArgument>()) {
    return false;
  }
  auto lhsLoad = dyn_cast<SPNBatchRead>(lhs.getDefiningOp());
  auto rhsLoad = dyn_cast<SPNBatchRead>(rhs.getDefiningOp());
  if (!lhsLoad || !rhsLoad) {
    return false;
  }
  if (lhsLoad.batchMem() != rhsLoad.batchMem()) {
    return false;
  }
  if (lhsLoad.batchIndex() != rhsLoad.batchIndex()) {
    return false;
  }
  if (lhsLoad.sampleIndex() + 1 != rhsLoad.sampleIndex()) {
    return false;
  }
  return true;
}

void slp::dumpSLPNode(SLPNode const& node) {
  for (size_t i = node.numVectors(); i-- > 0;) {
    dumpSLPNodeVector(*node.getVector(i));
  }
}

size_t slp::numNodes(SLPNode const& root) {
  return SLPNode::postOrder(root).size();
}
size_t slp::numVectors(SLPNode const& root) {
  size_t n = 0;
  for (auto* node : SLPNode::postOrder(root)) {
    n += node->numVectors();
  }
  return n;
}

// Helper functions in an anonymous namespace.
namespace {
  void dumpBlockArgOrDefiningAddress(Value const& val) {
    if (auto* definingOp = val.getDefiningOp()) {
      llvm::dbgs() << definingOp;
    } else {
      llvm::dbgs() << "block arg #" << val.cast<BlockArgument>().getArgNumber();
    }
  }
  void dumpBlockArgOrDefiningOpName(Value const& val) {
    if (auto* definingOp = val.getDefiningOp()) {
      llvm::dbgs() << definingOp->getName();
    } else {
      llvm::dbgs() << "block arg #" << val.cast<BlockArgument>().getArgNumber();
    }
  }
}

void slp::dumpSLPNodeVector(NodeVector const& nodeVector) {
  for (size_t lane = 0; lane < nodeVector.numLanes(); ++lane) {
    if (!nodeVector[lane].isa<BlockArgument>()) {
      llvm::dbgs() << nodeVector[lane] << " (" << nodeVector[lane].getDefiningOp() << ")";
    } else {
      dumpBlockArgOrDefiningOpName(nodeVector[lane]);
    }
    if (lane < nodeVector.numLanes() - 1) {
      llvm::dbgs() << "\t|\t";
    }
  }
  llvm::dbgs() << "\n";
}

void slp::dumpOpTree(ArrayRef<Value> const& values) {
  DenseMap<Value, unsigned> nodes;
  SmallVector<std::tuple<Value, Value, unsigned>> edges;

  std::vector<Value> worklist;
  for (auto const& value : values) {
    worklist.emplace_back(value);
  }

  while (!worklist.empty()) {
    auto value = worklist.back();
    worklist.pop_back();
    if (nodes.count(value)) {
      continue;
    }
    nodes[value] = nodes.size();
    if (auto* definingOp = value.getDefiningOp()) {
      for (unsigned i = 0; i < definingOp->getNumOperands(); ++i) {
        auto const& operand = definingOp->getOperand(i);
        edges.emplace_back(std::make_tuple(value, operand, i));
        worklist.emplace_back(operand);
      }
    }
  }

  llvm::dbgs() << "digraph debug_graph {\n";
  llvm::dbgs() << "rankdir = BT;\n";
  llvm::dbgs() << "node[shape=box];\n";
  for (auto const& entry : nodes) {
    auto const& value = entry.first;
    auto const& id = entry.second;
    llvm::dbgs() << "\tnode_" << id << "[label=\"";
    if (auto* definingOp = value.getDefiningOp()) {
      llvm::dbgs() << definingOp->getName().getStringRef() << "\\n" << definingOp;
      if (auto constantOp = dyn_cast<ConstantOp>(definingOp)) {
        if (constantOp.value().getType().isIntOrIndex()) {
          llvm::dbgs() << "\\nvalue: " << std::to_string(constantOp.value().dyn_cast<IntegerAttr>().getInt());
        } else if (constantOp.value().getType().isIntOrFloat()) {
          llvm::dbgs() << "\\nvalue: " << std::to_string(constantOp.value().dyn_cast<FloatAttr>().getValueAsDouble());
        }
      } else if (auto batchReadOp = dyn_cast<SPNBatchRead>(definingOp)) {
        llvm::dbgs() << "\\nbatch mem: " << batchReadOp.batchMem().dyn_cast<BlockArgument>().getArgNumber();
        llvm::dbgs() << "\\nbatch index: " << batchReadOp.batchMem().dyn_cast<BlockArgument>().getArgNumber();
        llvm::dbgs() << "\\nsample index: " << batchReadOp.sampleIndex();
      }
    } else {
      dumpBlockArgOrDefiningAddress(value);
    }
    llvm::dbgs() << "\", fillcolor=\"#a0522d\"];\n";
  }
  for (auto const& edge : edges) {
    llvm::dbgs() << "\tnode_" << nodes[std::get<0>(edge)] << " -> node_" << nodes[std::get<1>(edge)] << "[label=\""
                 << std::get<2>(edge) << "\"];\n";
  }
  llvm::dbgs() << "}\n";
}

void slp::dumpSLPGraph(SLPNode const& root) {

  llvm::dbgs() << "digraph debug_graph {\n";
  llvm::dbgs() << "rankdir = BT;\n";
  llvm::dbgs() << "node[shape=box];\n";

  std::queue<SLPNode const*> worklist;
  worklist.emplace(&root);

  while (!worklist.empty()) {
    auto const* node = worklist.front();
    worklist.pop();

    llvm::dbgs() << "node_" << node << "[label=<\n";
    llvm::dbgs() << "\t<TABLE ALIGN=\"CENTER\" BORDER=\"0\" CELLSPACING=\"10\" CELLPADDING=\"0\">\n";
    for (size_t i = node->numVectors(); i-- > 0;) {
      llvm::dbgs() << "\t\t<TR>\n";
      for (size_t lane = 0; lane < node->numLanes(); ++lane) {
        auto value = node->getValue(lane, i);
        llvm::dbgs() << "\t\t\t<TD>";
        llvm::dbgs() << "<B>";
        dumpBlockArgOrDefiningOpName(value);
        llvm::dbgs() << "</B>";
        // --- Additional operation information ---
        if (auto* definingOp = value.getDefiningOp()) {
          llvm::dbgs() << "<BR/><FONT COLOR=\"#bbbbbb\">";
          llvm::dbgs() << "(" << definingOp << ")";
          if (auto constOp = dyn_cast<ConstantOp>(definingOp)) {
            llvm::dbgs() << "<BR/>value: " << constOp.getValue();
          } else if (auto lowConstOp = dyn_cast<SPNConstant>(definingOp)) {
            llvm::dbgs() << "<BR/>value: " << lowConstOp.value().convertToDouble();
          } else if (auto readOp = dyn_cast<SPNBatchRead>(definingOp)) {
            llvm::dbgs() << "<BR/>mem: ";
            dumpBlockArgOrDefiningAddress(readOp.batchMem());
            llvm::dbgs() << "<BR/>batch: ";
            dumpBlockArgOrDefiningAddress(readOp.batchIndex());
            llvm::dbgs() << "<BR/>sample: " << readOp.sampleIndex();
          } else if (auto gaussianOp = dyn_cast<SPNGaussianLeaf>(definingOp)) {
            llvm::dbgs() << "<BR/>index: ";
            dumpBlockArgOrDefiningAddress(gaussianOp.index());
            llvm::dbgs() << "<BR/>mean: " << gaussianOp.mean().convertToDouble();
            llvm::dbgs() << "<BR/>stddev: " << gaussianOp.stddev().convertToDouble();
          } else if (auto categoricalOp = dyn_cast<SPNCategoricalLeaf>(definingOp)) {
            llvm::dbgs() << "<BR/>index: ";
            dumpBlockArgOrDefiningAddress(categoricalOp.index());
            llvm::dbgs() << "<BR/>probabilities: [ ";
            for (auto const& probability : categoricalOp.probabilities()) {
              llvm::dbgs() << probability << " ";
            }
            llvm::dbgs() << "]";
          }
          llvm::dbgs() << "</FONT>";
        }
        // --- ================================ ---
        llvm::dbgs() << "</TD>";
        if (lane < node->numLanes() - 1) {
          llvm::dbgs() << "<VR/>";
        }
        llvm::dbgs() << "\n";
      }
      llvm::dbgs() << "\t\t</TR>\n";
    }
    llvm::dbgs() << "\t</TABLE>\n";
    llvm::dbgs() << ">];\n";

    for (size_t i = 0; i < node->numOperands(); ++i) {
      auto const& operand = node->getOperand(i);
      llvm::dbgs() << "node_" << node << "->" << "node_" << operand << "[label=\"" << std::to_string(i) << "\"];\n";
      worklist.emplace(operand);
    }
  }
  llvm::dbgs() << "}\n";
}
