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

  std::vector<std::tuple<std::string, uint32_t>> args{
    {"--mm-data-width", 32},
    {"--mm-addr-width", 32},
    {"--stream-in-bytes", inputBitWidth / 8},
    {"--stream-out-bytes", outputBitWidth / 8},
    {"--pre-fifo-depth", preFifoDepth},
    {"--post-fifo-depth", postFifoDepth},
    {"--body-pipeline-depth", bodyDelay},
    {"--variable-count", variableCount},
    {"--bits-per-variable", bitsPerVariable},
    {"--body-output-width", bodyOutputWidth}
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

void EmbedController::fixAXISignalNames(mlir::ModuleOp op) {
  using namespace ::circt::hw;
  using PortInfo = ::circt::hw::PortInfo;

  static const std::string PREFIXES[] = {
    "M_AXI_"
  };

  OpBuilder builder(op.getContext());

  auto removeUnderscores = [](const std::string& name) -> std::string {
    return std::regex_replace(name, std::regex("_"), "");
  };

  op.walk([&](HWModuleOp modOp) {
    SmallVector<PortInfo> allPorts = modOp.getAllPorts();
    std::vector<std::pair<unsigned, PortInfo>> insertInputs, insertOutputs;
    std::vector<unsigned> eraseInputs, eraseOutputs;

    for (const PortInfo& portInfo : allPorts) {
      std::string name = portInfo.getName().str();
      std::string newName;
      bool found = false;

      for (const std::string& PREFIX : PREFIXES)
        if (name.find(PREFIX) == 0) {
          newName = removeUnderscores(name.substr(PREFIX.length()));
          found = true;
          break;
        }

      if (!found)
        continue;

      PortInfo newPortInfo = portInfo;
      newPortInfo.name = builder.getStringAttr(newName);
      unsigned portIndex = portInfo.argNum;

      if (portInfo.direction == PortDirection::INPUT) {
        insertInputs.emplace_back(portIndex, newPortInfo);
        eraseInputs.push_back(portIndex);
      } else {
        insertOutputs.emplace_back(portIndex, newPortInfo);
        eraseOutputs.push_back(portIndex);
      }
    }

    modOp.modifyPorts(
      insertInputs,
      insertOutputs,
      eraseInputs,
      eraseOutputs
    );
  });
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

  // TODO: Add signal renaming as pass
  fixAXISignalNames(op);

  op.dump();

  return failure("not implemented");
}

}