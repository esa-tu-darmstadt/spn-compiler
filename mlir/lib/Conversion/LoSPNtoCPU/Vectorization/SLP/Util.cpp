//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNtoCPU/Vectorization/SLP/Util.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::low;
using namespace mlir::spn::low::slp;

// TODO: I don't like this redefinition of default options. Maybe find a way to include GlobalOptions.h?
unsigned option::maxNodeSize = 10;
unsigned option::maxLookAhead = 3;
unsigned option::maxAttempts = 1;
unsigned option::maxSuccessfulIterations = 1;
bool option::reorderInstructionsDFS = true;
bool option::allowDuplicateElements = false;
bool option::allowTopologicalMixing = false;
bool option::useXorChains = false;

bool slp::vectorizable(Operation* op) {
  return (op->hasTrait<OpTrait::spn::low::VectorizableOp>() || op->hasTrait<OpTrait::ConstantLike>())
      && op->hasTrait<OpTrait::OneResult>() && ofVectorizableType(op->getResult(0));
}

bool slp::vectorizable(Value value) {
  if (auto* definingOp = value.getDefiningOp()) {
    if (!vectorizable(definingOp)) {
      return false;
    }
  }
  return ofVectorizableType(value);
}

bool slp::ofVectorizableType(Value value) {
  if (auto logType = value.getType().dyn_cast<LogType>()) {
    return VectorType::isValidElementType(logType.getBaseType());
  }
  return VectorType::isValidElementType(value.getType());
}

bool slp::commutative(Value value) {
  return value.getDefiningOp() && value.getDefiningOp()->hasTrait<OpTrait::IsCommutative>();
}

bool slp::consecutiveLoads(Value lhs, Value rhs) {
  if (lhs == rhs || !lhs.getDefiningOp() || !rhs.getDefiningOp()) {
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
  if (lhsLoad.dynamicIndex() != rhsLoad.dynamicIndex()) {
    return false;
  }
  return lhsLoad.staticIndex() + 1 == rhsLoad.staticIndex();
}

bool slp::anyGaussianMarginalized(Superword const& superword) {
  for (auto value : superword) {
    auto gaussianOp = value.getDefiningOp<SPNGaussianLeaf>();
    assert(gaussianOp && "only applicable to gaussian leaf vectors");
    if (gaussianOp.supportMarginal()) {
      return true;
    }
  }
  return false;
}

SmallVector<Value, 2> slp::getOperands(Value value) {
  SmallVector<Value, 2> operands;
  assert(value.getDefiningOp() && "operations without defining op do not have operands");
  operands.reserve(value.getDefiningOp()->getNumOperands());
  for (auto operand : value.getDefiningOp()->getOperands()) {
    operands.emplace_back(operand);
  }
  return operands;
}

void slp::sortByOpcode(SmallVectorImpl<Value>& values, Optional<OperationName> smallestOpcode) {
  llvm::sort(std::begin(values), std::end(values), [&](Value lhs, Value rhs) {
    auto* lhsOp = lhs.getDefiningOp();
    auto* rhsOp = rhs.getDefiningOp();
    if (!lhsOp && !rhsOp) {
      return lhs.cast<BlockArgument>().getArgNumber() < rhs.cast<BlockArgument>().getArgNumber();
    } else if (lhsOp && !rhsOp) {
      return true;
    } else if (!lhsOp && rhsOp) {
      return false;
    }
    if (smallestOpcode.hasValue()) {
      if (lhsOp->getName() == smallestOpcode.getValue()) {
        return rhsOp->getName() != smallestOpcode.getValue();
      } else if (rhsOp->getName() == smallestOpcode.getValue()) {
        return false;
      }
    }
    // Avoid tiebreaks.
    if (lhsOp->getName().getStringRef() == rhsOp->getName().getStringRef()) {
      return lhsOp->isBeforeInBlock(rhsOp);
    }
    return lhsOp->getName().getStringRef() < rhsOp->getName().getStringRef();
  });
}

// Helper functions in an anonymous namespace.
namespace {
  void dumpBlockArgOrDefiningAddress(Value val) {
    if (auto* definingOp = val.getDefiningOp()) {
      llvm::dbgs() << definingOp;
    } else {
      llvm::dbgs() << "block arg #" << val.cast<BlockArgument>().getArgNumber();
    }
  }
  void dumpBlockArgOrDefiningOpName(Value val) {
    if (auto* definingOp = val.getDefiningOp()) {
      llvm::dbgs() << definingOp->getName();
    } else {
      llvm::dbgs() << "block arg #" << val.cast<BlockArgument>().getArgNumber();
    }
  }
}

void slp::dumpSuperword(Superword const& superword) {
  for (size_t lane = 0; lane < superword.numLanes(); ++lane) {
    if (!superword[lane].isa<BlockArgument>()) {
      llvm::dbgs() << superword[lane] << " (" << superword[lane].getDefiningOp() << ")";
    } else {
      dumpBlockArgOrDefiningOpName(superword[lane]);
    }
    if (lane < superword.numLanes() - 1) {
      llvm::dbgs() << "\t|\t";
    }
  }
  llvm::dbgs() << "\n";
}

void slp::dumpSLPNode(SLPNode const& node) {
  for (size_t i = node.numSuperwords(); i-- > 0;) {
    dumpSuperword(*node.getSuperword(i));
  }
}

void slp::dumpOpGraph(ArrayRef<Value> values) {
  DenseMap<Value, unsigned> nodes;
  SmallVector<std::tuple<Value, Value, unsigned>> edges;

  std::vector<Value> worklist;
  for (auto value : values) {
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
        auto operand = definingOp->getOperand(i);
        edges.emplace_back(std::make_tuple(value, operand, i));
        worklist.emplace_back(operand);
      }
    }
  }

  llvm::dbgs() << "digraph debug_graph {\n";
  llvm::dbgs() << "rankdir = BT;\n";
  llvm::dbgs() << "node[shape=box];\n";
  for (auto const& entry : nodes) {
    auto value = entry.first;
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
        llvm::dbgs() << "\\dynamic index: " << batchReadOp.dynamicIndex();
        llvm::dbgs() << "\\nstatic index: " << batchReadOp.staticIndex();
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

// Helper functions in anonymous namespace.
namespace {
  void dumpAdditionalInformation(Value value) {
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
        dumpBlockArgOrDefiningAddress(readOp.dynamicIndex());
        llvm::dbgs() << "<BR/>sample: " << readOp.staticIndex();
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
  }
}

void slp::dumpSuperwordGraph(Superword* root) {

  llvm::dbgs() << "digraph debug_graph {\n";
  llvm::dbgs() << "rankdir = BT;\n";
  llvm::dbgs() << "node[shape=box];\n";

  for (auto* superword : graph::postOrder(root)) {
    llvm::dbgs() << "node_" << superword << "[label=<\n";
    llvm::dbgs() << "\t<TABLE ALIGN=\"CENTER\" BORDER=\"0\" CELLSPACING=\"10\" CELLPADDING=\"0\">\n";
    llvm::dbgs() << "\t\t<TR>\n";
    for (size_t lane = 0; lane < superword->numLanes(); ++lane) {
      auto value = superword->getElement(lane);
      llvm::dbgs() << "\t\t\t<TD>";
      llvm::dbgs() << "<B>";
      if (superword->hasAlteredSemanticsInLane(lane)) {
        llvm::dbgs() << "<FONT COLOR=\"crimson\">";
      }
      dumpBlockArgOrDefiningOpName(value);
      if (superword->hasAlteredSemanticsInLane(lane)) {
        llvm::dbgs() << "</FONT>";
      }
      llvm::dbgs() << "</B>";
      // --- Additional operation information ---
      dumpAdditionalInformation(value);
      // --- ================================ ---
      llvm::dbgs() << "</TD>";
      if (lane < superword->numLanes() - 1) {
        llvm::dbgs() << "<VR/>";
      }
      llvm::dbgs() << "\n";
    }
    llvm::dbgs() << "\t\t</TR>\n";
    llvm::dbgs() << "\t</TABLE>\n";
    llvm::dbgs() << ">];\n";

    for (size_t i = 0; i < superword->numOperands(); ++i) {
      auto* operand = superword->getOperand(i);
      llvm::dbgs() << "node_" << superword << "->" << "node_" << operand << "[label=\"" << std::to_string(i)
                   << "\"];\n";
    }
  }
  llvm::dbgs() << "}\n";
}

void slp::dumpSLPGraph(SLPNode* root, bool includeInputs) {

  llvm::dbgs() << "digraph debug_graph {\n";
  llvm::dbgs() << "rankdir = BT;\n";
  llvm::dbgs() << "node[shape=box];\n";

  for (auto* node : graph::postOrder(root)) {
    llvm::dbgs() << "node_" << node << "[label=<\n";
    llvm::dbgs() << "\t<TABLE ALIGN=\"CENTER\" BORDER=\"0\" CELLSPACING=\"10\" CELLPADDING=\"0\">\n";
    for (size_t i = node->numSuperwords(); i-- > 0;) {
      llvm::dbgs() << "\t\t<TR>\n";
      for (size_t lane = 0; lane < node->numLanes(); ++lane) {
        auto value = node->getValue(lane, i);
        llvm::dbgs() << "\t\t\t<TD>";
        llvm::dbgs() << "<B>";
        if (node->getSuperword(i)->hasAlteredSemanticsInLane(lane)) {
          llvm::dbgs() << "<FONT COLOR=\"crimson\">";
        }
        dumpBlockArgOrDefiningOpName(value);
        if (node->getSuperword(i)->hasAlteredSemanticsInLane(lane)) {
          llvm::dbgs() << "</FONT>";
        }
        llvm::dbgs() << "</B>";
        // --- Additional operation information ---
        dumpAdditionalInformation(value);
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

    if (node->numOperands() > 0) {
      for (size_t i = 0; i < node->numOperands(); ++i) {
        auto* operand = node->getOperand(i);
        llvm::dbgs() << "node_" << node << "->" << "node_" << operand << "[label=\"" << std::to_string(i) << "\"];\n";
      }
    } else if (includeInputs) {
      llvm::SmallPtrSet<Operation*, 8> inputs;
      for (size_t lane = 0; lane < node->numLanes(); ++lane) {
        auto element = node->getValue(lane, node->numSuperwords() - 1);
        if (auto* definingOp = element.getDefiningOp()) {
          for (unsigned n = 0; n < definingOp->getNumOperands(); ++n) {
            auto operand = definingOp->getOperand(n);
            if (auto* input = operand.getDefiningOp()) {
              if (inputs.insert(input).second) {
                llvm::dbgs() << "input_" << input << "[label=<\n";
                llvm::dbgs() << "\t<TABLE ALIGN=\"CENTER\" BORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"0\">\n";
                llvm::dbgs() << "\t\t<TR>\n";
                llvm::dbgs() << "\t\t\t<TD>";
                llvm::dbgs() << "<B>";
                dumpBlockArgOrDefiningOpName(operand);
                llvm::dbgs() << "</B>";
                // --- Additional operation information ---
                dumpAdditionalInformation(operand);
                // --- ================================ ---
                llvm::dbgs() << "</TD>";
                llvm::dbgs() << "</TR>\n";
                llvm::dbgs() << "\t</TABLE>\n";
                llvm::dbgs() << ">];\n";
              }
              llvm::dbgs() << "node_" << node << "->" << "input_" << input << "[label=\"" << lane << "." << n
                           << "\"];\n";
            }
          }
        }
      }
    }
  }
  llvm::dbgs() << "}\n";
}

void slp::dumpDependencyGraph(DependencyGraph const& dependencyGraph) {
  llvm::dbgs() << "digraph debug_graph {\n";
  llvm::dbgs() << "rankdir = TB;\n";
  llvm::dbgs() << "node[shape=box];\n";
  for (auto* node : dependencyGraph.nodes) {
    llvm::dbgs() << "node_" << node << "[label=<\n";
    llvm::dbgs() << "\t<TABLE ALIGN=\"CENTER\" BORDER=\"0\" CELLSPACING=\"10\" CELLPADDING=\"0\">\n";
    llvm::dbgs() << "\t\t<TR>\n";
    for (size_t lane = 0; lane < node->numLanes(); ++lane) {
      auto value = node->getElement(lane);
      llvm::dbgs() << "\t\t\t<TD>";
      llvm::dbgs() << "<B>";
      dumpBlockArgOrDefiningOpName(value);
      llvm::dbgs() << "</B>";
      // --- Additional operation information ---
      dumpAdditionalInformation(value);
      // --- ================================ ---
      llvm::dbgs() << "</TD>";
      if (lane < node->numLanes() - 1) {
        llvm::dbgs() << "<VR/>";
      }
      llvm::dbgs() << "\n";
    }
    llvm::dbgs() << "\t\t</TR>\n";
    llvm::dbgs() << "\t</TABLE>\n";
    llvm::dbgs() << ">];\n";
  }
  for (auto const& entry : dependencyGraph.dependencyEdges) {
    auto* src = entry.first;
    for (auto* dst : entry.second) {
      llvm::dbgs() << "node_" << src << "->" << "node_" << dst << ";\n";
    }
  }
  llvm::dbgs() << "}\n";
}
