//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPASS_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPASS_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "SLPGraph.h"

namespace mlir {
  namespace spn {

    struct SLPVectorizationPass : public PassWrapper<SLPVectorizationPass, OperationPass<FuncOp>> {

    public:

      explicit SLPVectorizationPass() = default;

    protected:
      void runOnOperation() override;

    private:

      void transform(slp::SLPGraph& graph);
      Operation* transform(slp::SLPNode& node, bool isRoot);

      /// For debugging purposes.
      static void printSPN(std::vector<Operation*> const& operations) {
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
          printOperation(op);
        }
        for (auto& edge : edges) {
          printEdge(std::get<0>(edge), std::get<1>(edge), std::get<2>(edge));
        }
        llvm::dbgs() << "}\n";
      }

      /// For debugging purposes.
      static void printOperation(Operation* op) {
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

    };

    std::unique_ptr<Pass> createSLPVectorizationPass();
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPASS_H
