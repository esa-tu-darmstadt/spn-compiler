//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPU/Vectorization/SLP/SLPNode.h"
#include "llvm/Support/Debug.h"
#include <queue>

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::low::slp;

SLPNode::SLPNode(std::vector<Operation*> const& operations) : lanes{operations.size()}, operandNodes{}, nodeInputs{} {
  for (size_t i = 0; i < operations.size(); ++i) {
    lanes[i].emplace_back(operations[i]);
  }
}

void SLPNode::addOperationToLane(Operation* operation, size_t const& lane) {
  lanes[lane].emplace_back(operation);
}

Operation* SLPNode::getOperation(size_t lane, size_t index) const {
  assert(lane <= numLanes() && index <= numVectors());
  return lanes[lane][index];
}

void SLPNode::setOperation(size_t lane, size_t index, Operation* operation) {
  assert(lane <= numLanes() && index <= numVectors());
  lanes[lane][index] = operation;
}

bool SLPNode::isMultiNode() const {
  return numVectors() > 1;
}

bool SLPNode::isUniform() const {
  return std::all_of(std::begin(lanes), std::end(lanes), [&](auto const& operations) {
    return operations[0]->getName() == lanes[0][0]->getName();
  });
}

bool SLPNode::areRootOfNode(std::vector<Operation*> const& operations) const {
  for (size_t lane = 0; lane < numLanes(); ++lane) {
    if (operations[lane] != lanes[lane].front()) {
      return false;
    }
  }
  return true;
}

size_t SLPNode::numLanes() const {
  return lanes.size();
}

size_t SLPNode::numVectors() const {
  return lanes.front().size();
}

std::vector<Operation*> SLPNode::getVector(size_t index) const {
  assert(index <= numVectors());
  std::vector<Operation*> vector;
  for (size_t lane = 0; lane < numLanes(); ++lane) {
    vector.emplace_back(lanes[lane][index]);
  }
  return vector;
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

Type SLPNode::getResultType() const {
  return lanes[0][0]->getResult(0).getType();
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
        llvm::dbgs() << "<B>" << operation->getName() << "</B> <FONT COLOR=\"#bbbbbb\">(" << operation << ")</FONT>";
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
