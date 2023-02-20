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
#include <regex>


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
  FPGAKernel *kernel = getContext()->get<FPGAKernel>();

  std::vector<std::tuple<std::string, uint32_t>> args{
    {"--memory-data-width", kernel->memDataWidth},
    {"--memory-addr-width", kernel->memAddrWidth},
    {"--stream-in-bytes", kernel->sAxisControllerWidth / 8},
    {"--stream-out-bytes", kernel->mAxisControllerWidth / 8},
    {"--fifo-depth", kernel->fifoDepth},
    {"--body-pipeline-depth", kernel->bodyDelay},
    {"--var-count", kernel->spnVarCount},
    {"--bits-per-var", kernel->spnBitsPerVar},
    {"--result-bit-width", kernel->spnResultWidth}
  };

  std::string cmdArgs;
  for (const auto& [param, value] : args)
    cmdArgs.append(std::string(" ") + param + " " + std::to_string(value));

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

  FPGAKernel kernel;
  getContext()->add<FPGAKernel>(std::move(kernel));
  FPGAKernel *pKernel = getContext()->get<FPGAKernel>();
  pKernel->bodyDelay = bodyDelay;
  pKernel->fifoDepth = bodyDelay * 2;
  pKernel->spnVarCount = kernelInfo->numFeatures;
  pKernel->spnBitsPerVar = 8; // TODO
  pKernel->spnResultWidth = 31; // TODO

  pKernel->mAxisControllerWidth = round8(pKernel->spnResultWidth);
  pKernel->sAxisControllerWidth = round8(pKernel->spnBitsPerVar * pKernel->spnVarCount);

  pKernel->memDataWidth = 32;
  pKernel->memAddrWidth = 32;

  pKernel->liteDataWidth = 32;
  pKernel->liteAddrWidth = 32;
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

bool EmbedController::fixAXISignalNames(mlir::ModuleOp op) {
  ConversionTarget target(*op.getContext());

  target.addLegalDialect<::circt::sv::SVDialect>();
  target.addDynamicallyLegalDialect<::circt::hw::HWDialect>(
    [](Operation *op) {
      auto modOp = llvm::dyn_cast<::circt::hw::HWModuleOp>(op);
      
      if (!modOp)
        return true;

      return AXI4SignalNameRewriting::containsUnfixedAXI4Names(modOp);  
    }
  );

  RewritePatternSet patterns(op.getContext());
  patterns.add<AXI4SignalNameRewriting>(op.getContext());

  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  
  return mlir::succeeded(
    applyPartialConversion(op, target, frozenPatterns)
  );
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

  if (!fixAXISignalNames(op))
    return failure("Fixing signal names failed");

  //op.dump();

  return success();
}

const std::string AXI4SignalNameRewriting::PREFIXES[] = {
  "M_AXI_",
  "S_AXI_LITE_"
};

LogicalResult AXI4SignalNameRewriting::matchAndRewrite(::circt::hw::HWModuleOp op,
                                                       OpAdaptor adaptor,
                                                       ConversionPatternRewriter& rewriter) const {
  using namespace ::circt::hw;
  using PortInfo = ::circt::hw::PortInfo;

  auto removeUnderscores = [](const std::string& name) -> std::string {
    return std::regex_replace(name, std::regex("_"), "");
  };

  SmallVector<PortInfo> allPorts = op.getAllPorts();
  std::vector<PortInfo> newPorts;

  for (const PortInfo& portInfo : allPorts) {
    std::string name = portInfo.getName().str();
    std::string newName;
    bool found = false;

    for (const std::string& prefix : PREFIXES)
      if (name.find(prefix) == 0) {
        newName = prefix + removeUnderscores(name.substr(prefix.length()));
        found = true;
        break;
      }

    PortInfo newPortInfo = portInfo;

    if (found)
      newPortInfo.name = rewriter.getStringAttr(newName);

    newPorts.push_back(newPortInfo);
  }

  ::circt::hw::HWModuleOp newOp = rewriter.replaceOpWithNewOp<::circt::hw::HWModuleOp>(op,
    op.getNameAttr(),
    newPorts
  );

  // TODO: I don't know if this is dangerous :S
  Region& newBody = newOp.getFunctionBody();
  Region& oldBody = op.getFunctionBody();
  newBody.takeBody(oldBody);

  return mlir::success();
}

bool AXI4SignalNameRewriting::containsUnfixedAXI4Names(::circt::hw::HWModuleOp op) {
  using namespace ::circt::hw;
  using PortInfo = ::circt::hw::PortInfo;

  static std::regex legal{ []() {
    size_t i = 0, n = sizeof(PREFIXES) / sizeof(PREFIXES[0]); std::string s;
    for (const std::string& prefix : PREFIXES) {
      bool isLast = i++ == n - 1;
      s = s + (prefix + "[a-zA-Z]+") + (isLast ? "" : "|");
    }
    return s;
  }() };

  for (const PortInfo& portInfo : op.getAllPorts()) {
    std::string name = portInfo.getName().str();
    //llvm::outs() << "Checking " << name << "\n";

    for (const std::string& prefix : PREFIXES)
      if (name.find(prefix) == 0 && !std::regex_match(name, legal)) {
        //llvm::outs() << "  Illegal!\n";
        return false;
      }
  }

  //llvm::outs() << "  Legal!\n";
  return true;
}

}