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

        void dump() const;

      private:

        void buildGraph(std::vector<Operation*> const& operations, node_t const& currentNode);

        void reorderOperands(node_t const& multinode);

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

        static std::vector<std::vector<Operation*>> getOperandsVectorized(std::vector<Operation*> const& operations) {
          for (auto* operation : operations) {
            assert(operation->getNumOperands() == operations.front()->getNumOperands()
                       && "operations must have same numbers of operands");
          }
          std::vector<std::vector<Operation*>> allOperands;
          for (size_t i = 0; i < operations.front()->getNumOperands(); ++i) {
            std::vector<Operation*> operands;
            operands.reserve(operations.size());
            for (auto* operation : operations) {
              operands.emplace_back(operation->getOperand(i).getDefiningOp());
            }
            allOperands.emplace_back(operands);
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

        std::map<node_t, std::vector<node_t>> operandsOf;

        size_t const maxLookAhead;

      };
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPGRAPH_H
