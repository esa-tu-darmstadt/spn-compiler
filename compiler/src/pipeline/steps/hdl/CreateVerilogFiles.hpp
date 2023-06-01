#pragma once

#include "Kernel.h"

#include "pipeline/steps/mlir/MLIRPassPipeline.h"

#include "pipeline/PipelineStep.h"
#include "toolchain/MLIRToolchain.h"
#include "mlir/IR/BuiltinOps.h"

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"

#include "circt/Dialect/FIRRTL/FIRParser.h"

#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESITypes.h"

#include "mlir/Transforms/DialectConversion.h"

#include <firp/AXIStream.hpp>
#include <firp/ShiftRegister.hpp>
#include <firp/firpQueue.hpp>

#include <filesystem>


namespace spnc {

struct CreateVerilogFilesConfig {
  std::filesystem::path targetDir;
  std::string tmpdirName;
  std::string topName;
};

class CreateVerilogFiles : public StepSingleInput<CreateVerilogFiles, mlir::ModuleOp>,
                           public StepWithResult<Kernel> {
  CreateVerilogFilesConfig config;
public:
  explicit CreateVerilogFiles(StepWithResult<mlir::ModuleOp>& root, const CreateVerilogFilesConfig& config):
    StepSingleInput<CreateVerilogFiles, mlir::ModuleOp>(root), config(config) {}

  ExecutionResult executeStep(mlir::ModuleOp *root);

  Kernel *result() override { return getContext()->get<Kernel>(); }

  STEP_NAME("create-verilog-files");
};

}