#pragma once

#include "pipeline/steps/mlir/MLIRPassPipeline.h"

#include "pipeline/PipelineStep.h"
#include "toolchain/MLIRToolchain.h"
#include "mlir/IR/BuiltinOps.h"

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"

#include "circt/Dialect/FIRRTL/FIRParser.h"

#include <filesystem>



namespace spnc {

// Call the scala app to generate the firrtl code for the controller.
// Load the firrtl file into Circt using the builtin parser.
// Lower the firrtl ops to hw ops.
// Replace the external spn body module in the AST with our own spn body module.
// Return the resulting hw AST.

namespace fs = std::filesystem;

struct ControllerConfig {
  fs::path generatorPath;
};

using namespace ::circt::hw;
using namespace ::mlir;
using namespace ::circt::firrtl;

struct GeneratorConfig {
  fs::path generatorPath;
  uint32_t
    inputBitWidth,
    outputBitWidth,
    bodyDelay,
    preFifoDepth,
    postFifoDepth;
};

/*
  TODO:
    - convert FIR code to HW code
    - replace the HW external op definition of the SPN body with the actual
      implementation
    - insert the categorical/histogram modules (and more) in the controller AST
 */

class EmbedController : public StepSingleInput<EmbedController, mlir::ModuleOp>,
                        public StepWithResult<mlir::ModuleOp> {
  ControllerConfig config;
  uint32_t
    inputBitWidth,
    outputBitWidth,
    bodyDelay,
    preFifoDepth,
    postFifoDepth;

  std::optional<mlir::ModuleOp> generateController(MLIRContext *ctxt);

  // fails if there is more than one spn_body
  std::optional<HWModuleOp> getUniqueBody(mlir::ModuleOp root);
  LogicalResult insertBodyIntoController(ModuleOp controller, HWModuleOp body);
  void setParameters(uint32_t bodyDelay);
public:
  explicit EmbedController(const ControllerConfig& config, StepWithResult<mlir::ModuleOp>& spnBody):
    StepSingleInput<EmbedController, mlir::ModuleOp>(spnBody),    
    config(config) {}

  ExecutionResult executeStep(mlir::ModuleOp *spnBody);

  mlir::ModuleOp *result() override { return topModule.get(); }

  STEP_NAME("embed-controller");
private:
  std::unique_ptr<mlir::ModuleOp> topModule;
};

}