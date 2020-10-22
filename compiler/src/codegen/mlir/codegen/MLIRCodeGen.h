//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_CODEGEN_MLIRCODEGEN_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_CODEGEN_MLIRCODEGEN_H

#include <graph-ir/IRGraph.h>
#include <mlir/IR/Module.h>
#include <driver/Actions.h>
#include <mlir/IR/Builders.h>
#include <unordered_map>

using namespace mlir;

namespace spnc {

  class MLIRCodeGen : public ActionSingleInput<IRGraph, ModuleOp> {

  public:

    MLIRCodeGen(ActionWithOutput<IRGraph>& _input,
                std::string _kernelName,
                std::shared_ptr<MLIRContext> _context);

    ModuleOp& execute() override;

  private:

    void generateMLIR(IRGraph& graph);

    std::string kernelName;

    std::shared_ptr<MLIRContext> context;

    mlir::OpBuilder builder;

    std::unique_ptr<ModuleOp> module;

    bool cached = false;

    std::unordered_map<std::string, mlir::Value> node2value;

  };

}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_CODEGEN_MLIRCODEGEN_H
