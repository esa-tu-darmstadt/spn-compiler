#include "EmbedController.hpp"

#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"

#include "circt/Conversion/FIRRTLToHW.h"

#include "llvm/Support/SourceMgr.h"
#include "mlir/Support/Timing.h"
#include "spdlog/fmt/fmt.h"
#include "util/Command.h"
#include <filesystem>


namespace spnc {

std::optional<ModuleOp> EmbedController::generateController(MLIRContext *ctxt) {
  // call scala CLI app
  std::string cmdArgs =
    fmt::format(" -in-width {} -out-width {} -body-depth {} -pre-fifo-depth {} -post-fifo-depth {}",
      inputBitWidth,
      outputBitWidth,
      bodyDelay,
      preFifoDepth,
      postFifoDepth
    );

  std::string cmdString = config.generatorPath.string() + cmdArgs;
  spdlog::info("Executing shell command: {}", cmdString);
  std::string result = spnc::Command::executeExternalCommandAndGetOutput(cmdString);
  auto sourceBuffer = llvm::MemoryBuffer::getMemBufferCopy(result);

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(sourceBuffer), llvm::SMLoc());

  TimingScope ts;

  mlir::OwningOpRef<mlir::ModuleOp> op = importFIRFile(sourceMgr, ctxt, ts, {});

  if (!op)
    return std::nullopt;

  return op.release();
}

std::optional<HWModuleOp> EmbedController::getUniqueBody(ModuleOp root) {
  uint32_t count = 0;
  HWModuleOp body;

  root.walk([&](HWModuleOp op) {
    // check name
    if (op.getName() == "spn_body") {
      body = op;
      ++count;
    }
  });

  if (count != 1)
    return std::nullopt;

  return body;
}

LogicalResult EmbedController::insertBodyIntoController(ModuleOp controller, HWModuleOp body) {
  // find the external module
  // instantiate hw instance with a reference to our body

  //controller.walk([&](::circt::firrtl::InstanceOp op) {
  //  if (op.getModuleName() != "ExternalSPNBody")
  //    return;
//
  //  auto refMod = op.getModule();
  //  refMod
  //});

  return mlir::failure();
}

void EmbedController::setParameters(uint32_t bodyDelay) {
  KernelInfo *kernelInfo = getContext()->get<KernelInfo>();

  inputBitWidth = kernelInfo->numFeatures * kernelInfo->bytesPerFeature * 8;
  outputBitWidth = kernelInfo->numResults * kernelInfo->bytesPerResult * 8;

  this->bodyDelay = bodyDelay;
  preFifoDepth = bodyDelay * 2;
  postFifoDepth = bodyDelay * 2;
}

ExecutionResult EmbedController::executeStep(ModuleOp *root) {
  std::optional<HWModuleOp> spnBody = getUniqueBody(*root);

  if (!spnBody.has_value())
    return failure(
      "EmbedController: spn_body must be unique."
    );

  Operation *op = spnBody.value().getOperation();

  if (!op->hasAttr("fpga.body_delay"))
    return failure(
      "EmbedController: spn_body does not have an attribute fpga.body_delay."
    );

  IntegerAttr attr = llvm::dyn_cast<IntegerAttr>(
    op->getAttr("fpga.body_delay")
  );

  if (!attr)
    return failure(
      "EmbedController: fpga.body_delay is not an integer."
    );

  setParameters(attr.getInt());

  std::optional<ModuleOp> controllerOpt = generateController(root->getContext());

  if (!controllerOpt.has_value())
    return failure(
      "EmbedController: generateController() has failed."
    );

  ModuleOp controller = controllerOpt.value();
  {
    ExecutionResult result = convertFirrtlToHw(controller);
    if (failed(result))
      return result;
  }

  controller.dump();

  //if (failed(insertBodyIntoController(firController.value(), spnBody.value())))
  //  return failure(
  //    "EmbedController: Could not insert body into controller."
  //  );

  return failure("EmbedController is not implemented!");
}

ExecutionResult EmbedController::convertFirrtlToHw(mlir::ModuleOp op) {
  MLIRContext *ctxt = op.getContext();
  PassManager pm(ctxt);

  pm.addPass(circt::createLowerFIRRTLToHWPass());

  //op.dump();

  if (failed(pm.run(op)))
    return failure("Converting FIRRTL to HW failed");

  if (failed(op.verify()))
    return failure("Verifying module after conversion failed");

  return success();
}

}