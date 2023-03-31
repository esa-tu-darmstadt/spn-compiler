#include "EmbedController.hpp"

#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"

#include "circt/Conversion/FIRRTLToHW.h"
#include "circt/Dialect/FIRRTL/Passes.h"

#include "circt/Dialect/Seq/SeqPasses.h"

#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"

#include "ReturnKernel.h"
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
  FPGAKernel& kernel = getContext()->get<Kernel>()->getFPGAKernel();

  std::vector<std::tuple<std::string, uint32_t>> args{
    {"--memory-data-width", kernel.memDataWidth},
    {"--memory-addr-width", kernel.memAddrWidth},
    {"--stream-in-bytes", kernel.sAxisControllerWidth / 8},
    {"--stream-out-bytes", kernel.mAxisControllerWidth / 8},
    {"--fifo-depth", kernel.fifoDepth},
    {"--body-pipeline-depth", kernel.bodyDelay},
    {"--var-count", kernel.spnVarCount},
    {"--bits-per-var", kernel.spnBitsPerVar},
    {"--result-bit-width", kernel.spnResultWidth}
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
  FPGAKernel& pKernel = getContext()->get<Kernel>()->getFPGAKernel();
  pKernel.bodyDelay = bodyDelay;
  pKernel.fifoDepth = bodyDelay * 2;
}

ModuleOp EmbedController::createController(MLIRContext *ctxt) {
  return ModuleOp();
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

  using namespace ::firp;
  using namespace ::firp::axis;

  FPGAKernel& kernel = getContext()->get<Kernel>()->getFPGAKernel();
  initFirpContext(*root, "SPNController");

  AXIStreamConfig slaveConfig{
    .dataBits = uint32_t(kernel.sAxisControllerWidth),
    .userBits = 1,
    .destBits = 1,
    .idBits = 1
  };

  AXIStreamConfig masterConfig{
    .dataBits =  uint32_t(kernel.mAxisControllerWidth),
    .userBits = 1,
    .destBits = 1,
    .idBits = 1
  };

  // Build a wrapper around 

  Controller controller(slaveConfig, masterConfig,
    kernel.spnVarCount, kernel.spnBitsPerVar, kernel.spnResultWidth, kernel.fifoDepth, kernel.bodyDelay);
  controller.makeTop();

  firpContext()->finish();
  firpContext()->verify();

  ExecutionResult result = convertFirrtlToHw(*root, spnBody.value());
  topModule = std::make_unique<mlir::ModuleOp>(*root);

  return result;

  /*

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
   */
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

ExecutionResult EmbedController::convertFirrtlToHw(mlir::ModuleOp op, circt::hw::HWModuleOp spnBody) {
  MLIRContext *ctxt = op.getContext();
  PassManager pm(ctxt);

  // inspired by firtool
  pm.addNestedPass<circt::firrtl::CircuitOp>(
    circt::firrtl::createInferWidthsPass()
  );
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

  // replace the external spn_body_external instance with an instance to the actual
  // spn_body
  struct RewriteInstance : public OpConversionPattern<circt::hw::InstanceOp> {
  private:
    Operation *modOp;
  public:
    RewriteInstance(MLIRContext *ctxt, Operation *modOp):
      OpConversionPattern(ctxt), modOp(modOp) {}

    LogicalResult matchAndRewrite(circt::hw::InstanceOp op,
                                  OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
      if (op.getModuleName() != "spn_body_external")
        return mlir::failure();

      std::vector<Value> inputs(adaptor.getInputs().begin(), adaptor.getInputs().end());

      rewriter.replaceOpWithNewOp<circt::hw::InstanceOp>(op,
        modOp,
        op.getName(),
        ArrayRef<Value>(inputs)
      );

      return mlir::success();
    }

    static bool isLegal(Operation *op) {
      auto instOp = llvm::dyn_cast<circt::hw::InstanceOp>(op);
      return !instOp || instOp.getModuleName() != "spn_body_external";
    }
  };

  ConversionTarget target(*ctxt);

  target.addLegalDialect<::circt::sv::SVDialect>();
  target.addDynamicallyLegalDialect<::circt::hw::HWDialect>(RewriteInstance::isLegal);

  RewritePatternSet patterns(ctxt);
  patterns.add<RewriteInstance>(ctxt, spnBody.getOperation());

  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  
  if (mlir::succeeded(applyPartialConversion(op, target, frozenPatterns)))
    return success();

  return failure("rewriting instances failed");
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

void SPNBody::body(uint32_t spnVarCount, uint32_t bitsPerVar, uint32_t spnResultWidth) {
  std::vector<Port> ports;

  for (size_t i = 0; i < spnVarCount; ++i)
    ports.push_back(
      Port("in_" + std::to_string(i), true, uintType(bitsPerVar))
    );

  ports.push_back(
    Port("out_prob", false, uintType(spnResultWidth))
  );

  ExternalSPNBody extBody(ports);

  for (size_t i = 0; i < spnVarCount; ++i)
    extBody.io("in_" + std::to_string(i)) <<= io("in")((i + 1) * bitsPerVar - 1, i * bitsPerVar);

  io("out") <<= extBody.io("out_prob");
}

void Controller::body(const AXIStreamConfig& slaveConfig, const AXIStreamConfig& masterConfig,
  uint32_t spnVarCount, uint32_t bitsPerVar, uint32_t resultWidth, uint32_t fifoDepth, uint32_t bodyDelay) {
  // TODO: reinsert
  //assert(kernel.bodyDelay <= kernel.fifoDepth);

  AXIStreamReceiver receiver(slaveConfig);
  receiver.io("AXIS") <<= io("AXIS_SLAVE");
  auto receiverDeqFire = Wire(doesFire(receiver.io("deq")), "receiverDeqFire");

  AXIStreamSender sender(masterConfig);
  io("AXIS_MASTER") <<= sender.io("AXIS");

  FirpQueue fifo(withLast(uintType(resultWidth)), fifoDepth);
  sender.io("enq") <<= fifo.io("deq");
  auto fifoEnqFire = Wire(doesFire(fifo.io("enq")), "fifoEnqFire");

  SPNBody spnBody(spnVarCount, bitsPerVar, resultWidth);
  // TODO: check endianness / order
  spnBody.io("in") <<= receiver.io("deq")("bits")("bits");

  // TODO: Build controller after scheduling!
  //size_t bodyDelay = fifoDepth;
  ShiftRegister lastDelayed(bitType(), bodyDelay);
  lastDelayed.io("in") <<= receiver.io("deq")("bits")("last");

  ShiftRegister validDelayed(bitType(), bodyDelay);
  validDelayed.io("in") <<= receiver.io("deq")("valid");

  fifo.io("enq")("bits")("bits") <<= spnBody.io("out");
  fifo.io("enq")("bits")("last") <<= lastDelayed.io("out");
  fifo.io("enq")("valid") <<= validDelayed.io("out");

  Reg itemCountInPipeline(uintType(16), "itemCountInPipeline");
  itemCountInPipeline.write(
    itemCountInPipeline.read()
    + mux(receiverDeqFire, cons(1), cons(0))
    - mux(fifoEnqFire, cons(1), cons(0))
  );

  auto canEnqueue = Wire(itemCountInPipeline.read() + fifo.io("count") + cons(2) <= cons(fifoDepth), "canEnqueue");
  receiver.io("deq")("ready") <<= canEnqueue;

  svCocoTBVerbatim(getName());
}

/*

Original source code:

class SPNController(config: SPNControllerConfig, modGen: SPNControllerConfig => SPN) extends Module {
  val slaveConfig = AXIStreamConfig(
    dataWidth = config.streamInBytes * 8,
    userWidth = 1,
    destWidth = 1,
    idWidth = 1
  )

  val masterConfig = AXIStreamConfig(
    dataWidth = config.streamOutBytes * 8,
    userWidth = 1,
    destWidth = 1,
    idWidth = 1
  )

  val AXI_SLAVE = IO(VerilogAXIStreamSlave(slaveConfig))
  val AXI_MASTER = IO(VerilogAXIStreamMaster(masterConfig))

  val receiver = Module(new AXIStreamReceiver(slaveConfig, 2))
  val sender = Module(new AXIStreamSender(masterConfig, 2))
  //val body = Module(new SPNBody(config))
  val body = modGen(config)
  body.clk := clock.asBool
  body.rst := reset.asBool
  // the actual fifo that buffers the results of spn body
  require(config.bodyPipelineDepth <= config.fifoDepth)
  val fifo = Module(new Queue[WithLast[UInt]](WithLast(UInt((config.streamOutBytes * 8).W)), 2))
  val itemCountInPipeline = RegInit(0.U(16.W))
  val canEnqueue = WireInit(2.U + itemCountInPipeline + fifo.io.count <= fifo.entries.U)

  receiver.AXI_SLAVE <> AXI_SLAVE
  sender.AXI_MASTER <> AXI_MASTER

  // the results can be moved into the body when there is a query and the fifo has enough capacity to buffer
  //body.io.in := receiver.out.bits.bits.asTypeOf(Vector(UInt(config.bitsPerVariable.W), config.variableCount))
  for (i <- 0 until config.varCount)
    body.in(i) := receiver.out.bits.bits((i + 1) * config.bitsPerVar - 1, i * config.bitsPerVar)

  // TODO: Logic loop with receiver.out.ready?
  receiver.out.ready := canEnqueue

  // connect spn body to the fifo (the fifo ready signal plays no role)
  fifo.io.enq.bits.bits := body.out.pad(fifo.io.enq.bits.bits.getWidth)
  fifo.io.enq.bits.last :=
    ShiftRegister(
      receiver.out.bits.last, config.bodyPipelineDepth,
      0.U, true.B
    )
  fifo.io.enq.valid := ShiftRegister(receiver.out.fire, config.bodyPipelineDepth, false.B, true.B)

  fifo.io.deq <> sender.in

  val receiverCounter = Counter(receiver.out.fire, 1000000)
  dontTouch(receiverCounter._1)
  val sendCounter = Counter(sender.in.fire, 1000000)
  dontTouch(sendCounter._1)

  // count how many items currently are in the spn body pipeline
  itemCountInPipeline := (
    itemCountInPipeline
    + Mux(receiver.out.fire, 1.U, 0.U)
    - Mux(fifo.io.enq.fire, 1.U, 0.U)
  )
}

*/

}
