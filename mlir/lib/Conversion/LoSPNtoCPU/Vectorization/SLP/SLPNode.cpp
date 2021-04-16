//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPU/Vectorization/SLP/SLPNode.h"
#include "LoSPN/LoSPNOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/Support/Debug.h"
#include <queue>

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::low::slp;

SLPNode::SLPNode(std::vector<Operation*> const& operations) {
  assert(!operations.empty());
  vectors.emplace_back(operations);
}

Operation* SLPNode::getOperation(size_t lane, size_t index) const {
  assert(lane <= numLanes() && index <= numVectors());
  return vectors[index][lane];
}

void SLPNode::setOperation(size_t lane, size_t index, Operation* operation) {
  assert(lane <= numLanes() && index <= numVectors());
  vectors[index][lane] = operation;
}

bool SLPNode::isMultiNode() const {
  return numVectors() > 1;
}

bool SLPNode::isUniform() const {
  return std::all_of(std::begin(vectors), std::end(vectors), [&](auto const& operations) {
    return operations[0]->getName() == vectors[0][0]->getName();
  });
}

bool SLPNode::containsOperation(Operation* op) const {
  return std::any_of(std::begin(vectors), std::end(vectors), [&](auto const& vectorOps) {
    return std::find(std::begin(vectorOps), std::end(vectorOps), op) != std::end(vectorOps);
  });
}

bool SLPNode::areRootOfNode(std::vector<Operation*> const& operations) const {
  return vectors[0] == operations;
}

size_t SLPNode::numLanes() const {
  return vectors[0].size();
}

size_t SLPNode::numVectors() const {
  return vectors.size();
}

void SLPNode::addVector(std::vector<Operation*> const& vectorOps) {
  assert(vectorOps.size() == numLanes());
  vectors.emplace_back(vectorOps);
}

std::vector<Operation*>& SLPNode::getVector(size_t index) {
  assert(index <= numVectors());
  return vectors[index];
}

std::vector<Operation*>& SLPNode::getVectorOf(Operation* op) {
  for (auto& vector : vectors) {
    if (std::find(std::begin(vector), std::end(vector), op) != std::end(vector)) {
      return vector;
    }
  }
  assert(false && "node does not contain the given operation");
}

std::vector<std::vector<Operation*>>& SLPNode::getVectors() {
  return vectors;
}

SLPNode* SLPNode::addOperand(std::vector<Operation*> const& operations) {
  operandNodes.emplace_back(std::make_unique<SLPNode>(operations));
  return operandNodes.back().get();
}

SLPNode* SLPNode::getOperand(size_t index) const {
  assert(index <= operandNodes.size());
  return operandNodes[index].get();
}

std::vector<SLPNode*> SLPNode::getOperands() const {
  std::vector<SLPNode*> operands;
  operands.reserve(operands.size());
  for (auto const& operand : operandNodes) {
    operands.emplace_back(operand.get());
  }
  return operands;
}

size_t SLPNode::numOperands() const {
  return operandNodes.size();
}

void SLPNode::addNodeInput(Value const& value) {
  nodeInputs.emplace_back(value);
}

Value const& SLPNode::getNodeInput(size_t index) const {
  return nodeInputs[index];
}

void SLPNode::dump() const {
  for (size_t i = numVectors(); i-- > 0;) {
    for (size_t lane = 0; lane < numLanes(); ++lane) {
      llvm::dbgs() << *getOperation(lane, i) << "(" << getOperation(lane, i) << ")";
      if (lane < numLanes() - 1) {
        llvm::dbgs() << "\t|\t";
      }
    }
    llvm::dbgs() << "\n";
  }
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
}

void SLPNode::dumpGraph() const {

  llvm::dbgs() << "digraph debug_graph {\n";
  llvm::dbgs() << "rankdir = BT;\n";
  llvm::dbgs() << "node[shape=box];\n";

  std::queue<SLPNode const*> worklist;
  worklist.emplace(this);

  while (!worklist.empty()) {
    auto const* node = worklist.front();
    worklist.pop();

    llvm::dbgs() << "node_" << node << "[label=<\n";
    llvm::dbgs() << "\t<TABLE ALIGN=\"CENTER\" BORDER=\"0\" CELLSPACING=\"10\" CELLPADDING=\"0\">\n";
    for (size_t i = node->numVectors(); i-- > 0;) {
      llvm::dbgs() << "\t\t<TR>\n";
      for (size_t lane = 0; lane < node->numLanes(); ++lane) {
        auto* operation = node->getOperation(lane, i);
        llvm::dbgs() << "\t\t\t<TD>";
        llvm::dbgs() << "<B>" << operation->getName() << "</B>";
        // --- Additional operation information ---
        llvm::dbgs() << "<BR/><FONT COLOR=\"#bbbbbb\">";
        llvm::dbgs() << "(" << operation << ")";
        if (auto constOp = dyn_cast<mlir::ConstantOp>(operation)) {
          llvm::dbgs() << "<BR/>value: " << constOp.getValue();
        } else if (auto lowConstOp = dyn_cast<low::SPNConstant>(operation)) {
          llvm::dbgs() << "<BR/>value: " << lowConstOp.value().convertToDouble();
        } else if (auto readOp = dyn_cast<low::SPNBatchRead>(operation)) {
          llvm::dbgs() << "<BR/>mem: ";
          dumpBlockArgOrDefiningAddress(readOp.batchMem());
          llvm::dbgs() << "<BR/>batch: ";
          dumpBlockArgOrDefiningAddress(readOp.batchIndex());
          llvm::dbgs() << "<BR/>sample: " << readOp.sampleIndex();
        }
        llvm::dbgs() << "</FONT>";
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

    for (auto const& operand : node->getOperands()) {
      llvm::dbgs() << "node_" << node << "->" << "node_" << operand << ";\n";
      worklist.emplace(operand);
    }
  }
  llvm::dbgs() << "}\n";
}
