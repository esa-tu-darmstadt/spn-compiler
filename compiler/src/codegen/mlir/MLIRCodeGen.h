//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_MLIRCODEGEN_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_MLIRCODEGEN_H

#include <codegen/mlir/dialects/spn/SPNDialect.h>
#include <driver/Actions.h>
#include <graph-ir/GraphIRNode.h>
#include <mlir/IR/Module.h>
#include <mlir/IR/Builders.h>
#include <unordered_map>
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace mlir::spn;

namespace spnc {

  class MLIRCodeGen : public ActionSingleInput<IRGraph, ModuleOp> {

  public:
    MLIRCodeGen(ActionWithOutput<IRGraph>& _input,
                const std::string& _kernelName,
                std::shared_ptr<MLIRContext> _context);

    ModuleOp& execute() override;

  private:

    void generateMLIR(IRGraph& graph);

    mlir::FuncOp createSPNFunction(uint32_t numInputs);

    std::shared_ptr<MLIRContext> context;

    mlir::OpBuilder builder;

    std::unique_ptr<ModuleOp> module;

    std::string kernelName;

    std::unordered_map<GraphIRNode*, mlir::Value> node2value;

    bool cached = false;

  };

}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_MLIRCODEGEN_H
