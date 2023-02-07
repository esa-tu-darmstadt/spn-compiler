#include "EmbedController.hpp"

#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"

#include "circt/Conversion/FIRRTLToHW.h"
#include "circt/Dialect/FIRRTL/Passes.h"

#include "circt/Dialect/Seq/SeqPasses.h"

#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"

#include "llvm/Support/SourceMgr.h"
#include "mlir/Support/Timing.h"
#include "spdlog/fmt/fmt.h"
#include "util/Command.h"
#include <filesystem>


namespace spnc {

void EmbedController::insertCocoTbDebug(ModuleOp controller, MLIRContext *ctxt) {
  OpBuilder builder(ctxt);

  const std::string debugString = R"(
`ifdef COCOTB_SIM
  initial begin
    $dumpfile("SPNController.vcd");
    $dumpvars (0, SPNController);
    #1;
  end
`endif
  )";
  
  controller.walk([&](FModuleOp modOp) {
    if (modOp.getName() != "SPNController")
      return;

    builder.setInsertionPointToEnd(modOp.getBodyBlock());
    builder.create<::circt::sv::VerbatimOp>(
      builder.getUnknownLoc(),
      debugString
    );
  });
}

std::optional<ModuleOp> EmbedController::generateController(MLIRContext *ctxt) {
  // call scala CLI app
  std::string cmdArgs =
    fmt::format(" -in-width {} -out-width {} -body-depth {} -pre-fifo-depth {} -post-fifo-depth {} -variable-count {} -bits-per-variable {} -body-output-width {}",
      inputBitWidth,
      outputBitWidth,
      bodyDelay,
      preFifoDepth,
      postFifoDepth,
      variableCount,
      bitsPerVariable,
      bodyOutputWidth
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

  ModuleOp controllerOp = op.release();
  insertCocoTbDebug(controllerOp, ctxt);

  return controllerOp;
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

LogicalResult EmbedController::insertBodyIntoController(ModuleOp controller, ModuleOp root, HWModuleOp spnBody) {
  // append all the operations at the end
  auto& dest = controller.getOperation()->getRegion(0).front().getOperations();
  auto& sourceOps = root.getOperation()->getRegion(0).front().getOperations();

  dest.splice(
    dest.end(),
    sourceOps
  );
  
  // find the reference to the external module
  Operation *refMod = nullptr;
  uint32_t refCount = 0;

  controller.walk([&](::circt::hw::InstanceOp op) {
    if (op.getModuleName() != "ExternalSPNBody")
      return;

    op.getModuleName();
    op.setModuleName(spnBody.moduleName());
    refMod = op.getReferencedModule();
    ++refCount;
  });

  assert(refMod);
  assert(refCount == 1);
  assert(refMod == spnBody.getOperation());

  return mlir::success();
}

void EmbedController::setParameters(uint32_t bodyDelay) {
  KernelInfo *kernelInfo = getContext()->get<KernelInfo>();

  inputBitWidth = kernelInfo->numFeatures * 8;
  outputBitWidth = 32;

  this->bodyDelay = bodyDelay;
  preFifoDepth = bodyDelay * 2;
  postFifoDepth = bodyDelay * 2;

  variableCount = kernelInfo->numFeatures;
  bitsPerVariable = 8;

  bodyOutputWidth = 31;
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

  if (failed(insertBodyIntoController(controller, *root, spnBody.value())))
    return failure(
      "EmbedController: Could not insert body into controller."
    );

  topModule = std::make_unique<mlir::ModuleOp>(controller);

  return success();
}

ExecutionResult EmbedController::convertFirrtlToHw(mlir::ModuleOp op) {
  MLIRContext *ctxt = op.getContext();
  PassManager pm(ctxt);

  // inspired by firtool
  pm.addNestedPass<circt::firrtl::CircuitOp>(
    circt::firrtl::createLowerFIRRTLTypesPass(
      circt::firrtl::PreserveAggregate::PreserveMode::None
    )
  );
  auto &modulePM = pm.nest<circt::firrtl::CircuitOp>().nest<circt::firrtl::FModuleOp>();
  modulePM.addPass(circt::firrtl::createExpandWhensPass());
  modulePM.addPass(circt::firrtl::createSFCCompatPass());  
  pm.addPass(circt::createLowerFIRRTLToHWPass());
  // export verilog doesn't know about seq.firreg
  pm.addNestedPass<circt::hw::HWModuleOp>(
    circt::seq::createSeqFIRRTLLowerToSVPass()
  );
  pm.addNestedPass<circt::hw::HWModuleOp>(
    circt::sv::createHWLegalizeModulesPass()
  );
  pm.addPass(circt::sv::createHWMemSimImplPass(
    //replSeqMem, ignoreReadEnableMem, stripMuxPragmas,
    //!isRandomEnabled(RandomKind::Mem), !isRandomEnabled(RandomKind::Reg),
    //addVivadoRAMAddressConflictSynthesisBugWorkaround
  ));

  // TODO: Add cleanup and canonicalization!

  if (failed(pm.run(op)))
    return failure("Converting FIRRTL to HW failed");

  if (failed(op.verify()))
    return failure("Verifying module after conversion failed");

  return success();
}

}