//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPGRAPH_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPGRAPH_H

#include "mlir/IR/Operation.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "LoSPN/LoSPNTraits.h"
#include "SLPMode.h"
#include "SLPNode.h"
#include "SLPSeeding.h"

#include <stack>

namespace mlir {
  namespace spn {
    namespace slp {

      class SLPGraph {

        typedef std::shared_ptr<SLPNode> node_t;

      public:

        SLPGraph(seed_t const& seed, size_t const& maxLookAhead);

        std::vector<std::shared_ptr<SLPNode>> getNodes() const;

      private:

        void buildGraph(std::vector<Operation*> const& operations, node_t const& currentNode);

        std::vector<std::vector<Operation*>> reorderOperands(node_t const& multinode);

        std::pair<Operation*, Mode> getBest(Mode const& mode,
                                            Operation* last,
                                            std::vector<Operation*>& candidates) const;

        int getLookAheadScore(Operation* last, Operation* candidate, size_t const& maxLevel) const;

        static bool areConsecutive(Operation* op1, Operation* op2) {
          auto load1 = dyn_cast<LoadOp>(op1);
          auto load2 = dyn_cast<LoadOp>(op2);
          if (!load1 || !load2) {
            return false;
          }
          if (load1.indices().size() != load2.indices().size()) {
            return false;
          }
          for (size_t i = 0; i < load1.indices().size(); ++i) {
            auto const1 = load1.indices()[i].getDefiningOp<ConstantOp>();
            auto const2 = load2.indices()[i].getDefiningOp<ConstantOp>();
            if (!const1 || !const2) {
              return false;
            }
            auto index1 = const1.value().dyn_cast<IntegerAttr>();
            auto index2 = const2.value().dyn_cast<IntegerAttr>();
            if (!index1 || !index2) {
              return false;
            }
            if (i == load1.indices().size() - 1) {
              return index1.getInt() == index2.getInt() - 1;
            } else if (index1.getInt() != index2.getInt()) {
              return false;
            }
          }
          return false;
        }

        /// Checks if the given operations are vectorizable.
        /// \param operations The potentially vectorizable operations.
        /// \return True if the operations can be vectorized, otherwise false.
        static bool vectorizable(std::vector<Operation*> const& operations) {
          for (size_t i = 0; i < operations.size(); ++i) {
            auto* op = operations[i];
            if (!op->hasTrait<OpTrait::OneResult>() || (i > 0 && op->getName() != operations.front()->getName())) {
              return false;
            }
          }
          return true;
        }

        static bool commutative(std::vector<Operation*> const& operations) {
          return std::all_of(std::begin(operations), std::end(operations), [&](Operation* op) {
            if (op->hasTrait<OpTrait::IsCommutative>()) {
              return true;
            }
            return dyn_cast<AddFOp>(op) || dyn_cast<MulFOp>(op);
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
          std::vector<Operation*> nodes;
          std::vector<std::tuple<Operation*, Operation*, size_t>> edges;

          std::stack<Operation*> worklist;
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
            printNode(op);
          }
          for (auto& edge : edges) {
            printEdge(std::get<0>(edge), std::get<1>(edge), std::get<2>(edge));
          }
          llvm::dbgs() << "}\n";
        }

        /// For debugging purposes.
        static void printNode(Operation* op) {
          llvm::dbgs() << "node_" << op << "[label=\"" << op->getName().getStringRef() << "\\n" << op;
          if (auto constantOp = dyn_cast<ConstantOp>(op)) {
            if (constantOp.value().getType().isIntOrIndex()) {
              llvm::dbgs() << "\\nvalue: " << std::to_string(constantOp.value().dyn_cast<IntegerAttr>().getInt());
            } else if (constantOp.value().getType().isIntOrFloat()) {
              llvm::dbgs() << "\\nvalue: "
                           << std::to_string(constantOp.value().dyn_cast<FloatAttr>().getValueAsDouble());
            }
          }
          llvm::dbgs() << "\", fillcolor=\"#a0522d\"];\n";
        }

        /// For debugging purposes.
        static void printEdge(Operation* src, Operation* dst, size_t index) {
          llvm::dbgs() << "node_" << src << " -> node_" << dst << "[label=\"" << std::to_string(index) << "\"];\n";
        }

        std::map<node_t, std::vector<node_t>> operandsOf;

        size_t const maxLookAhead;

      };
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPGRAPH_H
