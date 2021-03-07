//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_SPN_ANALYSIS_SLP_SLPGRAPH_H
#define SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_SPN_ANALYSIS_SLP_SLPGRAPH_H

#include "mlir/IR/Operation.h"
#include "SPN/SPNOpTraits.h"
#include "SLPMode.h"
#include "SLPNode.h"
#include "SLPSeeding.h"

#include <stack>

namespace mlir {
  namespace spn {
    namespace slp {

      class SLPTree {

        typedef std::shared_ptr<SLPNode> node_t;

      public:

        SLPTree(seed_t const& seed, size_t const& maxLookAhead);

      private:

        void buildGraph(std::vector<Operation*> const& operations, node_t const& currentNode);

        std::vector<std::vector<Operation*>> reorderOperands(node_t const& multinode);

        std::pair<Operation*, Mode> getBest(Mode const& mode,
                                            Operation* last,
                                            std::vector<Operation*>& candidates) const;

        int getLookAheadScore(Operation* last, Operation* candidate, size_t const& maxLevel) const;

        static bool areConsecutive(Operation* op1, Operation* op2) {
          if (auto leaf1 = dyn_cast<LeafNodeInterface>(op1)) {
            if (auto leaf2 = dyn_cast<LeafNodeInterface>(op2)) {
              return leaf2.getFeatureIndex() == leaf1.getFeatureIndex() + 1;
            }
          }
          return false;
        }

        /// Checks if the given operations are vectorizable. Operations are vectorizable iff the SPN dialect says they're
        /// vectorizable and they all share the same opcode.
        /// \param operations The potentially vectorizable operations.
        /// \return True if the operations can be vectorized, otherwise false.
        static bool vectorizable(std::vector<Operation*> const& operations) {
          for (size_t i = 0; i < operations.size(); ++i) {
            if (!operations.at(i)->hasTrait<OpTrait::spn::Vectorizable>()
                || (i > 0 && operations.at(i)->getName() != operations.front()->getName())) {
              return false;
            }
          }
          if (dyn_cast<GaussianOp>(operations.front())) {
            for (size_t i = 1; i < operations.size(); ++i) {
              if (!areConsecutive(operations.at(i - 1), operations.at(i))) {
                return false;
              }
            }
          }
          return true;
        }

        static bool commutative(std::vector<Operation*> const& operations) {
          return std::all_of(std::begin(operations), std::end(operations), [&](Operation* operation) {
            return operation->hasTrait<OpTrait::IsCommutative>();
          });
        }

        static bool escapesMultinode(Operation* operation) {
          // TODO: check if some intermediate, temporary value of a multinode is used outside of it
          assert(operation);
          return false;
        }

        static std::vector<Operation*> getOperands(Operation* operation) {
          std::vector<Operation*> operands;
          operands.reserve(operation->getNumOperands());
          for (auto operand : operation->getOperands()) {
            operands.emplace_back(operand.getDefiningOp());
          }
          return operands;
        }

        static std::vector<std::vector<Operation*>> getOperands(std::vector<Operation*> const& operations) {
          std::vector<std::vector<Operation*>> allOperands;
          allOperands.reserve(operations.size());
          for (auto* operation : operations) {
            allOperands.emplace_back(getOperands(operation));
          }
          return allOperands;
        }

        static std::vector<std::vector<Operation*>> getOperandsTransposed(std::vector<Operation*> const& operations) {
          std::vector<std::vector<Operation*>> allOperands;
          for (size_t i = 0; i < operations.front()->getNumOperands(); ++i) {
            std::vector<Operation*> operand;
            operand.reserve(operations.size());
            for (auto* operation : operations) {
              operand.emplace_back(operation->getOperand(i).getDefiningOp());
            }
            allOperands.emplace_back(operand);
          }
          return allOperands;
        }

        static void sortByOpcode(std::vector<Operation*>& operations,
                                 Optional<OperationName> const& smallestOpcode) {
          std::sort(std::begin(operations), std::end(operations), [&](Operation* lhs, Operation* rhs) {
            if (smallestOpcode.hasValue()) {
              if (lhs->getName() == smallestOpcode.getValue()) {
                return rhs->getName() != smallestOpcode.getValue();
              } else if (rhs->getName() == smallestOpcode.getValue()) {
                return false;
              }
            }
            return lhs->getName().getStringRef() < rhs->getName().getStringRef();
          });
        }

        /// For debugging purposes.
        static void printSubgraph(std::vector<Operation*> const& operations) {
          std::set<Operation*> nodes;
          std::vector<std::tuple<Operation*, Operation*, size_t>> edges;

          std::stack<Operation*> worklist;
          for (auto& op : operations) {
            worklist.push(op);
          }

          while (!worklist.empty()) {

            auto op = worklist.top();
            worklist.pop();

            if (nodes.find(op) == nodes.end()) {
              nodes.insert(op);
              for (size_t i = 0; i < op->getNumOperands(); ++i) {
                auto const& operand = op->getOperand(i);
                if (operand.getDefiningOp() != nullptr) {
                  edges.emplace_back(std::make_tuple(op, operand.getDefiningOp(), i));
                  worklist.push(operand.getDefiningOp());
                }
              }
            }
          }

          llvm::errs() << "digraph debug_graph {\n";
          llvm::errs() << "rankdir = BT;\n";
          llvm::errs() << "node[shape=box];\n";
          for (auto& op : nodes) {
            llvm::errs() << "node_" << op << "[label=\"" << op->getName().getStringRef() << "\\n" << op;
            if (auto leaf = dyn_cast<LeafNodeInterface>(op)) {
              llvm::errs() << "\\narg: " << leaf.getFeatureIndex() << "\\n";
            }
            llvm::errs() << "\", fillcolor=\"#a0522d\"];\n";
          }
          for (auto& edge : edges) {
            llvm::errs() << "node_" << std::get<0>(edge) << " -> node_" << std::get<1>(edge) << "[label=\""
                         << std::get<2>(edge) << "\"];\n";
          }
          llvm::errs() << "}\n";
        }

        std::map<node_t, std::vector<node_t>> operandsOf;

        size_t const maxLookAhead;

      };
    }
  }
}

#endif //SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_SPN_ANALYSIS_SLP_SLPGRAPH_H
