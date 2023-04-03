#pragma once

#include "Kernel.h"

#include "pipeline/steps/mlir/MLIRPassPipeline.h"

#include "pipeline/PipelineStep.h"
#include "toolchain/MLIRToolchain.h"
#include "mlir/IR/BuiltinOps.h"

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"

#include "circt/Dialect/FIRRTL/FIRParser.h"

#include "mlir/Transforms/DialectConversion.h"

#include "AXIStream.hpp"
#include "ShiftRegister.hpp"
#include "firpQueue.hpp"

#include <filesystem>



namespace spnc {

// Call the scala app to generate the firrtl code for the controller.
// Load the firrtl file into Circt using the builtin parser.
// Lower the firrtl ops to hw ops.
// Replace the external spn body module in the AST with our own spn body module.
// Return the resulting hw AST.

namespace fs = std::filesystem;

template <class T>
T round8(const T& n) {
  if (n % T(8) == 0)
    return n;
  return n + (T(8) - n % T(8));
}

struct ControllerConfig {
  fs::path generatorPath;
};

using namespace ::circt::hw;
using namespace ::mlir;
using namespace ::circt::firrtl;

class EmbedController : public StepSingleInput<EmbedController, mlir::ModuleOp>,
                        public StepWithResult<mlir::ModuleOp> {
  ControllerConfig config;

  // fails if there is more than one spn_body
  std::optional<HWModuleOp> getUniqueBody(mlir::ModuleOp root);
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

  static ExecutionResult convertFirrtlToHw(mlir::ModuleOp op, circt::hw::HWModuleOp spnBody);
};

using namespace firp;
using namespace firp::axis;

class ExternalSPNBody : public ExternalModule<ExternalSPNBody> {
public:
  ExternalSPNBody(const std::vector<Port>& ports):
    ExternalModule<ExternalSPNBody>(
      "spn_body_external",
      ports
    ) {}
};

class SPNBody : public Module<SPNBody> {
public:
  SPNBody(uint32_t spnVarCount, uint32_t bitsPerVar, uint32_t spnResultWidth):
    Module<SPNBody>(
      "SPNBody",
      {
        Port("in", true, uintType(spnVarCount * bitsPerVar)),
        Port("out", false, uintType(spnResultWidth))
      },
      spnVarCount, bitsPerVar, spnResultWidth
    ) {}

  void body(uint32_t spnVarCount, uint32_t bitsPerVar, uint32_t spnResultWidth);
};

class Controller : public Module<Controller> {
public:
  Controller(const AXIStreamConfig& slaveConfig, const AXIStreamConfig& masterConfig,
    uint32_t spnVarCount, uint32_t bitsPerVar, uint32_t resultWidth, uint32_t fifoDepth, uint32_t bodyDelay):
    Module<Controller>(
      "SPNController",
      {
        Port("AXIS_SLAVE", true, AXIStreamBundleType(slaveConfig)),
        Port("AXIS_MASTER", false, AXIStreamBundleType(masterConfig))
      },
      slaveConfig, masterConfig, spnVarCount, bitsPerVar, resultWidth, fifoDepth, bodyDelay
    ) {}

  void body(const AXIStreamConfig& slaveConfig, const AXIStreamConfig& masterConfig,
    uint32_t spnVarCount, uint32_t bitsPerVar, uint32_t resultWidth, uint32_t fifoDepth, uint32_t bodyDelay);
};

class ReplaceSPNBodyExternalPass {

};

}
