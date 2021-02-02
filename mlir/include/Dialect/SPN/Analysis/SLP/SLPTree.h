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

        std::vector<std::vector<node_t>> reorderOperands(node_t const& multinode);

        std::pair<Optional<node_t>, Mode> getBest(Mode const& mode,
                                                  node_t const& last,
                                                  std::vector<node_t>& candidates) const;

        int getLookAheadScore(node_t const& last, node_t const& candidate, size_t const& maxLevel) const;

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
          return true;
        }

        static bool commutative(std::vector<Operation*> const& operations) {
          return std::all_of(std::begin(operations), std::end(operations), [&](Operation* operation) {
            return operation->hasTrait<OpTrait::IsCommutative>();
          });
        }

        static bool escapesMultinode(Operation* operation) {
          // TODO: check if some intermediate, temporary value of a multinode is used outside of it
          return false;
        }

        static std::vector<std::vector<Operation*>> getOperands(std::vector<Operation*> const& operations) {
          std::vector<std::vector<Operation*>> allOperands;
          for (auto* operation : operations) {
            std::vector<Operation*> operands;
            operands.reserve(operation->getNumOperands());
            for (auto operand : operation->getOperands()) {
              operands.emplace_back(operand.getDefiningOp());
            }
            allOperands.emplace_back(operands);
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
              if (lhs->getName() == *smallestOpcode) {
                return rhs->getName() != *smallestOpcode;
              } else if (rhs->getName() == *smallestOpcode) {
                return false;
              }
            }
            return lhs->getName().getStringRef() < rhs->getName().getStringRef();
          });
        }

        /// For debugging purposes.
        static void printSubgraph(std::vector<Operation*> const& operations) {
          std::set<Operation*> nodes;
          std::vector<std::pair<Operation*, Operation*>> edges;

          std::stack<Operation*> worklist;
          for (auto& op : operations) {
            worklist.push(op);
          }

          while (!worklist.empty()) {

            auto op = worklist.top();
            worklist.pop();

            if (nodes.find(op) == nodes.end()) {
              nodes.insert(op);
              for (auto operand : op->getOperands()) {
                if (operand.getDefiningOp() != nullptr) {
                  edges.emplace_back(std::make_pair(op, operand.getDefiningOp()));
                  worklist.push(operand.getDefiningOp());
                }
              }
            }
          }

          llvm::errs() << "digraph debug_graph {\n";
          llvm::errs() << "rankdir = BT;\n";
          llvm::errs() << "node[shape=box];\n";
          for (auto& op : nodes) {
            llvm::errs() << "node_" << op << "[label=\"" << op->getName().getStringRef() << "\\n" << op << "\"];\n";
          }
          for (auto& edge : edges) {
            llvm::errs() << "node_" << edge.first << " -> node_" << edge.second << ";\n";
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
