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

void EmbedController::setParameters(uint32_t bodyDelay) {
  FPGAKernel& pKernel = getContext()->get<Kernel>()->getFPGAKernel();
  pKernel.bodyDelay = bodyDelay;
  pKernel.fifoDepth = bodyDelay * 2;
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
  initFirpContext(*root, "CosimTop");

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

  //Controller controller(slaveConfig, masterConfig,
  //  kernel.spnVarCount, kernel.spnBitsPerVar, kernel.spnResultWidth, kernel.fifoDepth, kernel.bodyDelay);
  //ReadyValidController controller(kernel.spnVarCount, kernel.spnBitsPerVar, kernel.spnResultWidth, kernel.fifoDepth, kernel.bodyDelay);
  //controller.makeTop();
  CosimTop top(kernel.spnVarCount, kernel.spnBitsPerVar, kernel.spnResultWidth, kernel.fifoDepth, kernel.bodyDelay);
  top.makeTop();

  firpContext()->finish();
  //firpContext()->verify();
  firpContext()->dump();

  ExecutionResult result = convertFirrtlToHw(*root, spnBody.value());

  //insertCosimTopLevel(*root, kernel.spnVarCount, kernel.spnBitsPerVar, kernel.spnResultWidth);


  topModule = std::make_unique<mlir::ModuleOp>(*root);

  topModule->dump();

  return result;
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

void EmbedController::insertCosimTopLevel(mlir::ModuleOp root, uint32_t spnVarCount, uint32_t bitsPerVar, uint32_t resultBitWidth) {
  // ReadyValidController

  MLIRContext *context = root.getContext();
  OpBuilder builder(context);

  // set insertion point
  builder.setInsertionPointToEnd(&root.getOperation()->getRegion(0).front());

  // helper functions
  auto strAttr = [&](const std::string& s) {
    return builder.getStringAttr(s);
  };

  auto portInfo = [&](const std::string& name, bool isInput, Type type) {
    return circt::hw::PortInfo{
      .name = strAttr(name),
      .direction = isInput ? circt::hw::PortDirection::INPUT : circt::hw::PortDirection::OUTPUT,
      type
    };
  };

  Type hwInputType = esi::lowerFIRRTLType(withLast(uintType(spnVarCount * bitsPerVar)));

  // define sender/receiver external ops
  auto senderChannelType = circt::esi::ChannelType::get(
    context,
    // this is actually withLast(uintType(...))
    // How do we predictably map this to HW types?
    hwInputType //,
    //circt::hw::StructType::get(context, ArrayRef<>)
  );

  std::vector<circt::hw::PortInfo> senderPorts{
    portInfo("chan", false, senderChannelType)
  };

  HWModuleExternOp mySender = builder.create<HWModuleExternOp>(
    builder.getUnknownLoc(),
    strAttr("MySender"),
    ArrayRef<circt::hw::PortInfo>(senderPorts)
  );

  auto receiverChannelType = circt::esi::ChannelType::get(
    context,
    builder.getIntegerType(spnVarCount * bitsPerVar)
  );

  std::vector<circt::hw::PortInfo> receiverPorts{
    portInfo("chan", true, receiverChannelType)
  };

  HWModuleExternOp myReceiver = builder.create<HWModuleExternOp>(
    builder.getUnknownLoc(),
    strAttr("MyReceiver"),
    ArrayRef<circt::hw::PortInfo>(receiverPorts)
  );

  // create actual module
  std::vector<circt::hw::PortInfo> ports{
    circt::hw::PortInfo{.name = strAttr("clock"), .direction = circt::hw::PortDirection::INPUT, .type = builder.getI1Type()},
    circt::hw::PortInfo{.name = strAttr("reset"), .direction = circt::hw::PortDirection::INPUT, .type = builder.getI1Type()}
  };

  HWModuleOp hwModOp = builder.create<HWModuleOp>(
    builder.getUnknownLoc(),
    builder.getStringAttr("top"),
    ArrayRef<circt::hw::PortInfo>(ports)
  );

  // set insertion point
  builder.setInsertionPointToStart(hwModOp.getBodyBlock());

  Value clock = builder.getBlock()->getArgument(0);
  Value reset = builder.getBlock()->getArgument(1);

  // instantiate sender/receiver
  circt::hw::InstanceOp mySenderInstance = builder.create<circt::hw::InstanceOp>(
    builder.getUnknownLoc(),
    mySender,
    strAttr("MySenderInstance"),
    ArrayRef<Value>()
  );

  // define the endpoints
  circt::esi::CosimEndpointOp ep = builder.create<circt::esi::CosimEndpointOp>(
    builder.getUnknownLoc(),
    receiverChannelType,
    clock, reset,
    mySenderInstance.getResult(0),
    strAttr("FuckingFunnyEP")
  );

  circt::hw::InstanceOp myReceiverInstance = builder.create<circt::hw::InstanceOp>(
    builder.getUnknownLoc(),
    myReceiver,
    strAttr("MyReceiverInstance"),
    ArrayRef<Value>(std::vector<Value>{ep.getOperation()->getResult(0)})
  );

  Value temporary = builder.create<circt::hw::ConstantOp>(
    builder.getUnknownLoc(),
    builder.getIntegerAttr(builder.getIntegerType(1), 1)
  ).getResult();

  // expose the ready-valid interfaces of the channels
  auto unwrapOp = builder.create<circt::esi::UnwrapValidReadyOp>(
    builder.getUnknownLoc(),
    mySenderInstance.getResult(0),
    temporary
  );

  /*
  Value rawData = unwrapOp.getResult(0);
  Value valid = unwrapOp.getResult(1);

  Value bits = builder.create<circt::hw::StructExtractOp>(
    builder.getUnknownLoc(),
    rawData,
    strAttr("bits")
  ).getResult();

  Value last = builder.create<circt::hw::StructExtractOp>(
    builder.getUnknownLoc(),
    rawData,
    strAttr("last")
  ).getResult();

  Value ready = builder.create<circt::hw::ConstantOp>(
    builder.getUnknownLoc(),
    builder.getIntegerAttr(builder.getIntegerType(1), 1)
  );

  // instantiate ReadyValidController
  std::vector<Value> inputs{
    clock, reset,
    valid, bits, last,
    ready
  };

  Operation *controller = nullptr;
  root.walk([&](HWModuleOp op){
    if (op.getName() == "ReadyValidController")
      controller = op.getOperation();
  });

  circt::hw::InstanceOp rvControllerInstance = builder.create<circt::hw::InstanceOp>(
    builder.getUnknownLoc(),
    controller,
    strAttr("ReadyValidControllerInstance"),
    ArrayRef<Value>(inputs)
  );

  Value enqReady = rvControllerInstance.getResult(0);
  Value deqValid = rvControllerInstance.getResult(1);
  Value deqBitsBits = rvControllerInstance.getResult(2);
  Value deqBitsLast = rvControllerInstance.getResult(3);

  Value deqBits = builder.create<circt::hw::StructCreateOp>(

  ).getResult();

  auto wrapOp = builder.create<circt::esi::UnwrapValidReadyOp>(
    builder.getUnknownLoc(),
    myReceiverInstance.getResult(0),
    deqBits, deqValid
  );*/
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
  auto receiverDeqFire = wireInit(doesFire(receiver.io("deq")), "receiverDeqFire");

  AXIStreamSender sender(masterConfig);
  io("AXIS_MASTER") <<= sender.io("AXIS");

  FirpQueue fifo(withLast(uintType(resultWidth)), fifoDepth);
  sender.io("enq") <<= fifo.io("deq");
  auto fifoEnqFire = wireInit(doesFire(fifo.io("enq")), "fifoEnqFire");

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

  auto canEnqueue = wireInit(itemCountInPipeline.read() + fifo.io("count") + cons(2) <= cons(fifoDepth), "canEnqueue");
  receiver.io("deq")("ready") <<= canEnqueue;

  svCocoTBVerbatim(getName());
}

void ReadyValidController::body(uint32_t spnVarCount, uint32_t bitsPerVar, uint32_t resultWidth, uint32_t fifoDepth, uint32_t bodyDelay) {
  auto enqFire = wireInit(doesFire(io("enq")), "enqFire");

  FirpQueue fifo(withLast(uintType(resultWidth)), fifoDepth);
  io("deq") <<= fifo.io("deq");
  auto deqFire = wireInit(doesFire(fifo.io("enq")), "deqFire");

  SPNBody spnBody(spnVarCount, bitsPerVar, resultWidth);
  spnBody.io("in") <<= io("enq")("bits")("bits");

  ShiftRegister lastDelayed(bitType(), bodyDelay);
  lastDelayed.io("in") <<= io("enq")("bits")("last");

  ShiftRegister validDelayed(bitType(), bodyDelay);
  validDelayed.io("in") <<= io("enq")("valid");

  fifo.io("enq")("bits")("bits") <<= spnBody.io("out");
  fifo.io("enq")("bits")("last") <<= lastDelayed.io("out");
  fifo.io("enq")("valid") <<= validDelayed.io("out");

  Reg itemCountInPipeline(uintType(16), "itemCountInPipeline");
  itemCountInPipeline.write(
    itemCountInPipeline.read()
    + mux(enqFire, cons(1), cons(0))
    - mux(deqFire, cons(1), cons(0))
  );

  auto canEnqueue = wireInit(itemCountInPipeline.read() + fifo.io("count") + cons(2) <= cons(fifoDepth), "canEnqueue");
  io("enq")("ready") <<= canEnqueue;

  svCocoTBVerbatim(getName());
}

void CosimTop::body(uint32_t spnVarCount, uint32_t bitsPerVar, uint32_t resultWidth, uint32_t fifoDepth, uint32_t bodyDelay) {

  ReadyValidController controller(spnVarCount, bitsPerVar, resultWidth, fifoDepth, bodyDelay);

  Receiver receiver(withLast(uintType(32)));

  // need external receiver
  // need external sender



  firpContext()->builder().create<circt::hw::ConstantOp>(
    firpContext()->builder().getUnknownLoc(),
    firpContext()->builder().getIntegerAttr(firpContext()->builder().getIntegerType(1), 1)
  ).getResult();
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

/*
  Roadmap:
  1. Wrap the datapath currently with AXI stream endpoints into a module with ESI endpoints.
    - Do we return a top level module with 2 ESI channels?
    - Implementation:
      - we have our data path module with clk, rst, 2 ESI channels
      - build a top module just with clk, rst and instantiate our data path
      - instantiate two ESI cosim endpoints  
  2. Run a simulation with ESI Cosim.
    - Our HW Code with the endpoints is compiled to verilog and put into a test bench. But how
      does it interface with the RPC CapnProto server?
    - How do we communicate with the RPC server?
  3. Extend this to the full controller.
    - How does it work with TaPaScO? Does it replace it?
  4. Think about AXI in context of ESI.

  Notes:
    - We seem to be able (theoretically) to lift clk and rst into the ESI-space via ESIPureModuleInputOp.



  Testing stuff:

  module {

    hw.module.extern @i1Fifo0(%in: !esi.channel<i16, FIFO0>) -> (out: !esi.channel<i16, FIFO0>)

    hw.module.extern @something(%in: !esi.channel<i16, ValidReady>) -> (out: !esi.channel<i16, ValidReady>)

    // by default this is also ValidReady
    hw.module.extern @something2(%in: !esi.channel<i16>) -> (out: !esi.channel<i16>)


    hw.module.extern @SenderHaha() -> (x: !esi.channel<si14>)
    hw.module.extern @RecieverLol(%a: !esi.channel<i32>)
    hw.module.extern @ArrRecieverLalala(%x: !esi.channel<!hw.array<4xsi64>>)

    hw.module @top(%clk:i1, %rst:i1) -> () {
      hw.instance "recv" @RecieverLol (a: %cosimRecv: !esi.channel<i32>) -> ()
      %send.x = hw.instance "send" @SenderHaha () -> (x: !esi.channel<si14>)
      %cosimRecv = esi.cosim %clk, %rst, %send.x, "TestEP" : !esi.channel<si14> -> !esi.channel<i32>
      %send2.x = hw.instance "send2" @SenderHaha () -> (x: !esi.channel<si14>)
      %cosimArrRecv = esi.cosim %clk, %rst, %send2.x, "ArrTestEP" : !esi.channel<si14> -> !esi.channel<!hw.array<4xsi64>>
      hw.instance "arrRecv" @ArrRecieverLalala (x: %cosimArrRecv: !esi.channel<!hw.array<4 x si64>>) -> ()
    }

  }

  ../../circt/build/bin/circt-opt --esi-emit-collateral=schema-file=s.capnp  --lower-esi-to-physical --lower-esi-ports --lower-esi-to-hw --export-verilog $1

 */

}
