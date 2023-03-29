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

  static void insertCocoTbDebug(ModuleOp controller, MLIRContext *ctxt);
  std::optional<mlir::ModuleOp> generateController(MLIRContext *ctxt);

  // fails if there is more than one spn_body
  std::optional<HWModuleOp> getUniqueBody(mlir::ModuleOp root);
  LogicalResult insertBodyIntoController(ModuleOp controller, ModuleOp root, HWModuleOp spnBody);
  void setParameters(uint32_t bodyDelay);

  ModuleOp createController(MLIRContext *ctxt);
public:
  explicit EmbedController(const ControllerConfig& config, StepWithResult<mlir::ModuleOp>& spnBody):
    StepSingleInput<EmbedController, mlir::ModuleOp>(spnBody),    
    config(config) {}

  ExecutionResult executeStep(mlir::ModuleOp *spnBody);

  mlir::ModuleOp *result() override { return topModule.get(); }

  STEP_NAME("embed-controller");
private:
  std::unique_ptr<mlir::ModuleOp> topModule;

  static bool fixAXISignalNames(mlir::ModuleOp op);
  static ExecutionResult convertFirrtlToHw(mlir::ModuleOp op);
};

class AXI4SignalNameRewriting : public OpConversionPattern<::circt::hw::HWModuleOp> {
  static const std::string PREFIXES[];
public:
  using OpConversionPattern<::circt::hw::HWModuleOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(::circt::hw::HWModuleOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override;

  static bool containsUnfixedAXI4Names(::circt::hw::HWModuleOp op);
};

using namespace firp;
using namespace firp::axis;

class SPNBodyPlaceholder : public Module<SPNBodyPlaceholder> {
public:
  SPNBodyPlaceholder(uint32_t inputWidth, uint32_t resultWidth):
    Module<SPNBodyPlaceholder>(
      "SPNBodyPlaceholder",
      {
        Port("in", true, uintType(inputWidth)),
        Port("out", false, uintType(resultWidth))
      },
      inputWidth, resultWidth
    ) {}

  void body(uint32_t inputWidth, uint32_t resultWidth) {
    io("out") <<= cons(0, uintType(resultWidth));
  }
};

class Controller : public Module<Controller> {
public:
  Controller(const AXIStreamConfig& slaveConfig, const AXIStreamConfig& masterConfig,
    uint32_t inputWidth, uint32_t resultWidth, uint32_t fifoDepth):
    Module<Controller>(
      "Controller",
      {
        Port("SLAVE", true, AXIStreamBundleType(slaveConfig)),
        Port("MASTER", false, AXIStreamBundleType(masterConfig))
      },
      slaveConfig, masterConfig, inputWidth, resultWidth, fifoDepth
    ) {}

  void body(const AXIStreamConfig& slaveConfig, const AXIStreamConfig& masterConfig,
    uint32_t inputWidth, uint32_t resultWidth, uint32_t fifoDepth);
};

}