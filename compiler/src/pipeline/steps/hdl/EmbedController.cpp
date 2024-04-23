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

Operation *EmbedController::implementESIreceiver(ModuleOp *root, uint32_t varCount, uint32_t bitsPerVar) {
  // create the HW implementation
  using namespace circt::hw;

  // TODO
  OpBuilder builder(root->getContext());
  builder.setInsertionPointToEnd(
    &root->getBodyRegion().back()
  );

  // build the receiver
  SmallVector<circt::hw::PortInfo> recvPorts{
    {.name = builder.getStringAttr("clk"), .direction = PortDirection::INPUT, .type = builder.getI1Type()},
    {.name = builder.getStringAttr("rst"), .direction = PortDirection::INPUT, .type = builder.getI1Type()},
    {.name = builder.getStringAttr("ready"), .direction = PortDirection::INPUT, .type = builder.getI1Type()},
    {.name = builder.getStringAttr("valid"), .direction = PortDirection::OUTPUT, .type = builder.getI1Type()},
    {.name = builder.getStringAttr("last"), .direction = PortDirection::OUTPUT, .type = builder.getI1Type()}
    // TODO: fix bit width
    //{.name = builder.getStringAttr("deq_bits"), .direction = PortDirection::OUTPUT, .type = builder.getIntegerType(41)}
  };

  for (uint32_t i = 0; i < varCount; ++i)
    recvPorts.push_back({
      .name = builder.getStringAttr("v" + std::to_string(i)), .direction = PortDirection::OUTPUT, .type = builder.getIntegerType(bitsPerVar)
    });

  circt::hw::HWModuleOp recvOp = builder.create<circt::hw::HWModuleOp>(
    builder.getUnknownLoc(),
    builder.getStringAttr("ESIReceiverLol"), recvPorts
  );

  // descend into the body and construct it
  builder.setInsertionPointToStart(recvOp.getBodyBlock());
  ESIReceiver::constructBody(builder, varCount, bitsPerVar);

  builder.getBlock()->back().erase();

  return recvOp.getOperation();
}

Operation *EmbedController::implementESIsender(ModuleOp *root) {
  // create the HW implementation
  using namespace circt::hw;

  // TODO
  OpBuilder builder(root->getContext());
  builder.setInsertionPointToEnd(
    &root->getBodyRegion().back()
  );

  // build the sender
  SmallVector<circt::hw::PortInfo> sendPorts{
    {.name = builder.getStringAttr("clk"), .direction = PortDirection::INPUT, .type = builder.getI1Type()},
    {.name = builder.getStringAttr("rst"), .direction = PortDirection::INPUT, .type = builder.getI1Type()},
    {.name = builder.getStringAttr("ready"), .direction = PortDirection::OUTPUT, .type = builder.getI1Type()},
    {.name = builder.getStringAttr("valid"), .direction = PortDirection::INPUT, .type = builder.getI1Type()},
    // TODO: fix bit width
    //{.name = builder.getStringAttr("enq_bits"), .direction = PortDirection::INPUT, .type = builder.getIntegerType(65)}
    {.name = builder.getStringAttr("last"), .direction = PortDirection::INPUT, .type = builder.getIntegerType(1)},
    {.name = builder.getStringAttr("data"), .direction = PortDirection::INPUT, .type = builder.getIntegerType(64)}
  };

  circt::hw::HWModuleOp sendOp = builder.create<circt::hw::HWModuleOp>(
    builder.getUnknownLoc(),
    builder.getStringAttr("ESISenderLol"), sendPorts
  );

  // descend into the body and construct it
  builder.setInsertionPointToStart(sendOp.getBodyBlock());
  ESISender::constructBody(builder);

  builder.getBlock()->back().erase();

  return sendOp.getOperation();
}

Operation *EmbedController::implementESIEndpoint(ModuleOp *root, uint32_t varCount, uint32_t bitsPerVar) {
  // create the HW implementation
  using namespace circt::hw;

  // TODO
  OpBuilder builder(root->getContext());
  builder.setInsertionPointToEnd(
    &root->getBodyRegion().back()
  );

  // a lot of ports...
  SmallVector<circt::hw::PortInfo> ports{
    {.name = builder.getStringAttr("clk"), .direction = PortDirection::INPUT, .type = builder.getI1Type()},
    {.name = builder.getStringAttr("rst"), .direction = PortDirection::INPUT, .type = builder.getI1Type()},
    {.name = builder.getStringAttr("varReady"), .direction = PortDirection::INPUT, .type = builder.getI1Type()},
    {.name = builder.getStringAttr("probValid"), .direction = PortDirection::INPUT, .type = builder.getI1Type()},
    {.name = builder.getStringAttr("probLast"), .direction = PortDirection::INPUT, .type = builder.getI1Type()},
    {.name = builder.getStringAttr("probData"), .direction = PortDirection::INPUT, .type = builder.getI64Type()}, // TODO: fix bit width

    {.name = builder.getStringAttr("varValid"), .direction = PortDirection::OUTPUT, .type = builder.getI1Type()},
    {.name = builder.getStringAttr("varLast"), .direction = PortDirection::OUTPUT, .type = builder.getI1Type()}
  };

  for (uint32_t i = 0; i < varCount; ++i)
    ports.push_back({
      .name = builder.getStringAttr("v" + std::to_string(i)), .direction = PortDirection::OUTPUT, .type = builder.getIntegerType(bitsPerVar)
    });

  ports.push_back(
    {.name = builder.getStringAttr("probReady"), .direction = PortDirection::OUTPUT, .type = builder.getI1Type()}
  );

  circt::hw::HWModuleOp recvOp = builder.create<circt::hw::HWModuleOp>(
    builder.getUnknownLoc(),
    builder.getStringAttr("ESIEndpointAbc"),
    ports
  );

  // descend into the body and construct it
  builder.setInsertionPointToStart(recvOp.getBodyBlock());
  ESIEndpoint::constructBody(builder, varCount, bitsPerVar);

  builder.getBlock()->back().erase();

  return recvOp.getOperation();
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
  attachFirpContext(*root, "top");
  firpContext()->defaultClockName = "clk";
  firpContext()->defaultResetName = "rst";

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
  {
    CosimTop top(kernel.spnVarCount, kernel.spnBitsPerVar, kernel.spnResultWidth, kernel.fifoDepth, kernel.bodyDelay);
    top.makeTop();
  }

  firpContext()->finish();
  //firpContext()->verify();
  //firpContext()->dump();

  ExecutionResult result = convertFirrtlToHw(*root, spnBody.value());

  //insertCosimTopLevel(*root, kernel.spnVarCount, kernel.spnBitsPerVar, kernel.spnResultWidth);

  Operation *recvModOp = nullptr; //implementESIreceiver(root, kernel.spnVarCount, kernel.spnBitsPerVar);
  Operation *sendModOp = nullptr; //implementESIsender(root);

  Operation *epModOp = implementESIEndpoint(root, kernel.spnVarCount, kernel.spnBitsPerVar);

  OpBuilder builder(root->getContext());

  // TODO: Big problem: Ints with more than 64 bits are not supported!

  // Find the instances of ESISender/ESIReceiver and replace them with instances pointing to
  // the newly generated implementation.
  if (false) {
    Operation *esiReceiverInstance = nullptr;

    // find the old receiver instance
    root->walk([&](circt::hw::InstanceOp op) {
      if (op.getInstanceName() == "ESIReceiver_instance")
        esiReceiverInstance = op.getOperation();
    });

    assert(esiReceiverInstance && "could not find instance");

    builder.setInsertionPointAfter(esiReceiverInstance);

    std::vector<Value> operands(
      esiReceiverInstance->operand_begin(),
      esiReceiverInstance->operand_end()
    );

    auto newEsiReceiverInstance = builder.create<circt::hw::InstanceOp>(
      builder.getUnknownLoc(),
      recvModOp,
      builder.getStringAttr("ESIReceiver_instance"),
      operands
    );

    esiReceiverInstance->replaceAllUsesWith(newEsiReceiverInstance.getResults());
    esiReceiverInstance->erase();
  }

  if (false) {
    Operation *esiSenderInstance = nullptr;

    // find the old receiver instance
    root->walk([&](circt::hw::InstanceOp op) {
      if (op.getInstanceName() == "ESISender_instance")
        esiSenderInstance = op.getOperation();
    });

    assert(esiSenderInstance && "could not find instance");

    builder.setInsertionPointAfter(esiSenderInstance);

    std::vector<Value> operands(
      esiSenderInstance->operand_begin(),
      esiSenderInstance->operand_end()
    );

    auto newEsiSenderInstance = builder.create<circt::hw::InstanceOp>(
      builder.getUnknownLoc(),
      sendModOp,
      builder.getStringAttr("ESISender_instance"),
      operands
    );

    esiSenderInstance->replaceAllUsesWith(newEsiSenderInstance.getResults());
    esiSenderInstance->erase();
  }

  if (true) {
    Operation *esiEndpointInstance = nullptr;

    // find the old receiver instance
    root->walk([&](circt::hw::InstanceOp op) {
      if (op.getInstanceName() == "ESIEndpoint_instance")
        esiEndpointInstance = op.getOperation();
    });

    assert(esiEndpointInstance && "could not find instance");

    builder.setInsertionPointAfter(esiEndpointInstance);

    std::vector<Value> operands(
      esiEndpointInstance->operand_begin(),
      esiEndpointInstance->operand_end()
    );

    auto newEsiEndpointInstance = builder.create<circt::hw::InstanceOp>(
      builder.getUnknownLoc(),
      epModOp,
      builder.getStringAttr("ESIEndpoint_instance"),
      operands
    );

    esiEndpointInstance->replaceAllUsesWith(newEsiEndpointInstance.getResults());
    esiEndpointInstance->erase();
  }  



  topModule = std::make_unique<mlir::ModuleOp>(*root);

  {
    std::error_code ec;
    llvm::raw_fd_ostream outFile("output.mlir", ec);
    topModule->print(outFile);
    //topModule->dump();
  }

  return result;
}

ExecutionResult EmbedController::convertFirrtlToHw(mlir::ModuleOp op, circt::hw::HWModuleOp spnBody) {
  MLIRContext *ctxt = op.getContext();
  PassManager pm(ctxt);

  // inspired by firtool
  
  pm.addNestedPass<circt::firrtl::CircuitOp>(
    circt::firrtl::createInferWidthsPass()
  );

  // this pass cannot be applied selectively
  pm.addNestedPass<circt::firrtl::CircuitOp>(
    circt::firrtl::createLowerFIRRTLTypesPass(
      // mode
      circt::firrtl::PreserveAggregate::PreserveMode::None,
      // memory mode
      circt::firrtl::PreserveAggregate::PreserveMode::None
    )
  );

  auto &modulePM = pm.nest<circt::firrtl::CircuitOp>().nest<circt::firrtl::FModuleOp>();
  modulePM.addPass(circt::firrtl::createExpandWhensPass());
  
  //modulePM.addPass(circt::firrtl::createSFCCompatPass());

  //pm.addNestedPass<mlir::ModuleOp>(
  //  circt::createLowerFIRRTLToHWPass()
  //);

  pm.addPass(
    circt::createLowerFIRRTLToHWPass()
  );


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
        op.getInstanceName(),
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

void SPNBody::body() {
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

void SPNDummyBody::body() {
  io("out") <<= cons(123, uintType(spnResultWidth));
}

void Controller::body() {
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

void ReadyValidController::body() {
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

void CosimTop::body() {
  /*
  ESIReceiver receiver(spnVarCount, bitsPerVar);
  ESISender sender(resultWidth);

  ReadyValidController controller(spnVarCount, bitsPerVar, resultWidth, fifoDepth, bodyDelay);

  controller.io("enq")("valid") <<= receiver.io("valid");
  
  std::vector<FValue> varValues;
  for (uint32_t i = 0; i < spnVarCount; ++i)
    varValues.push_back(receiver.io("v" + std::to_string(i)));
  controller.io("enq")("bits")("bits") <<= cat(varValues);

  controller.io("enq")("bits")("last") <<= receiver.io("last");
  receiver.io("ready") <<= controller.io("enq")("ready");

  sender.io("valid") <<= controller.io("deq")("valid");
  sender.io("last") <<= controller.io("deq")("bits")("last");
  sender.io("data") <<= controller.io("deq")("bits")("bits");
  controller.io("deq")("ready") <<= sender.io("ready");
   */

  ESIEndpoint endpoint(resultWidth, spnVarCount, bitsPerVar);
  ReadyValidController controller(spnVarCount, bitsPerVar, resultWidth, fifoDepth, bodyDelay);

  controller.io("enq")("valid") <<= endpoint.io("varValid");
  
  std::vector<FValue> varValues;
  for (uint32_t i = 0; i < spnVarCount; ++i)
    varValues.push_back(endpoint.io("v" + std::to_string(i)));
  controller.io("enq")("bits")("bits") <<= cat(varValues);

  controller.io("enq")("bits")("last") <<= endpoint.io("varLast");
  endpoint.io("varReady") <<= controller.io("enq")("ready");

  endpoint.io("probValid") <<= controller.io("deq")("valid");
  endpoint.io("probLast") <<= controller.io("deq")("bits")("last");
  endpoint.io("probData") <<= controller.io("deq")("bits")("bits");
  controller.io("deq")("ready") <<= endpoint.io("probReady");
}

void ESIReceiver::constructBody(OpBuilder builder, uint32_t varCount, uint32_t bitsPerVar) {
  // assume we are constructing the body of the HW module here

  /*
    hw.module @ESIReceiver(%clk: i1, %rst: i1, %deq_ready: i1) -> (deq_bits: i8, deq_valid: i1) {
      %nullChannel = esi.null : !esi.channel<i1>
      %cosimRecv = esi.cosim %clk, %rst, %nullChannel, "MyEP" : !esi.channel<i1> -> !esi.channel<i8>
      %bits, %valid = esi.unwrap.vr %cosimRecv, %deq_ready : i8
      hw.output %bits, %valid : i8, i1
    }
   */

  using namespace circt::esi;

  // build the struct type we receive
  using FieldInfo = circt::hw::detail::FieldInfo;
  std::vector<FieldInfo> structFieldInfos{
    FieldInfo{
      .name = builder.getStringAttr("last"),
      .type = builder.getI1Type()
    }
  };

  for (uint32_t i = 0; i < varCount; ++i)
    structFieldInfos.push_back(FieldInfo{
      .name = builder.getStringAttr("v" + std::to_string(i)),
      .type = builder.getIntegerType(bitsPerVar)
    });

  Type structType = circt::hw::StructType::get(builder.getContext(), structFieldInfos);

  // input arguments to the module
  Value clk, rst, deqReady;

  Block *block = builder.getBlock();
  clk = block->getArgument(0);
  rst = block->getArgument(1);
  deqReady = block->getArgument(2);

  // build a null channel
  Value nullChannel = builder.create<NullSourceOp>(
    builder.getUnknownLoc(),
    ChannelType::get(builder.getContext(), builder.getI1Type()) // cannot have 0 width for some reason
  ).getResult();

  // endpoint
  Value cosimRecv = builder.create<CosimEndpointOp>(
    builder.getUnknownLoc(),
    ChannelType::get(builder.getContext(), structType),
    clk, rst, nullChannel, builder.getStringAttr("MyReceiver")      
  ).getResult();

  // unwrap
  UnwrapValidReadyOp urvOp = builder.create<UnwrapValidReadyOp>(
    builder.getUnknownLoc(),
    cosimRecv, deqReady
  );

  Value rawOutput = urvOp.getResult(0);
  Value valid = urvOp.getResult(1);

  // decompose the struct and output it
  SmallVector<Value, 16> outputs{
    valid
  };

  outputs.push_back(
    builder.create<circt::hw::StructExtractOp>(
      builder.getUnknownLoc(),
      rawOutput,
      builder.getStringAttr("last")
    ).getResult()
  );

  for (uint32_t i = 0; i < varCount; ++i)
    outputs.push_back(
      builder.create<circt::hw::StructExtractOp>(
        builder.getUnknownLoc(),
        rawOutput,
        builder.getStringAttr("v" + std::to_string(i))
      ).getResult()
    );

  // output the results
  builder.create<circt::hw::OutputOp>(
    builder.getUnknownLoc(),
    outputs
  );
}

void ESISender::constructBody(OpBuilder builder) {
  using namespace circt::esi;

  // build the struct type we send
  using FieldInfo = circt::hw::detail::FieldInfo;
  std::vector<FieldInfo> structFieldInfos{
    FieldInfo{
      .name = builder.getStringAttr("last"),
      .type = builder.getI1Type()
    },
    FieldInfo{
      .name = builder.getStringAttr("data"),
      .type = builder.getI64Type()
    }
  };

  Type structType = circt::hw::StructType::get(builder.getContext(), structFieldInfos);

  // input arguments to the module
  Value clk, rst, last, data, enqValid;

  Block *block = builder.getBlock();
  clk = block->getArgument(0);
  rst = block->getArgument(1);
  enqValid = block->getArgument(2);
  last = block->getArgument(3);
  data = block->getArgument(4);

  // bundle the inputs into a struct
  Value actualInput = builder.create<circt::hw::StructCreateOp>(
    builder.getUnknownLoc(),
    structType,
    std::vector<Value>{last, data}
  ).getResult();

  // wrap
  WrapValidReadyOp wrvOp = builder.create<WrapValidReadyOp>(
    builder.getUnknownLoc(),
    actualInput, enqValid
  );

  Value sendChannel = wrvOp.getResult(0);
  Value enqReady = wrvOp.getResult(1);

  // endpoint
  Value cosimRecv = builder.create<CosimEndpointOp>(
    builder.getUnknownLoc(),
    // result type is something like { bits: i40, last: i1 }
    // TODO: generalize bit width for any SPN
    ChannelType::get(builder.getContext(), structType),
    clk, rst, sendChannel, builder.getStringAttr("MySender")      
  ).getResult();

  SmallVector<Value, 1> outputs{
    enqReady
  };

  // output the results
  builder.create<circt::hw::OutputOp>(
    builder.getUnknownLoc(),
    outputs
  );
}

void ESIEndpoint::constructBody(OpBuilder builder, uint32_t varCount, uint32_t bitsPerVar) {
  using namespace circt::esi;

  // build the struct types
  using FieldInfo = circt::hw::detail::FieldInfo;
  std::vector<FieldInfo> probStructFieldInfos{
    FieldInfo{
      .name = builder.getStringAttr("last"),
      .type = builder.getI1Type()
    },
    FieldInfo{
      .name = builder.getStringAttr("data"),
      .type = builder.getI64Type()
    }
  };

  Type probStructType = circt::hw::StructType::get(builder.getContext(), probStructFieldInfos);

  std::vector<FieldInfo> varStructFieldInfos{
    FieldInfo{
      .name = builder.getStringAttr("last"),
      .type = builder.getI1Type()
    }
  };

  for (uint32_t i = 0; i < varCount; ++i)
    varStructFieldInfos.push_back(FieldInfo{
      .name = builder.getStringAttr("v" + std::to_string(i)),
      .type = builder.getIntegerType(bitsPerVar)
    });

  Type varStructType = circt::hw::StructType::get(builder.getContext(), varStructFieldInfos);

  // input arguments to the module
  Block *block = builder.getBlock();
  Value clk = block->getArgument(0);
  Value rst = block->getArgument(1);
  Value varReady = block->getArgument(2);
  Value probValid = block->getArgument(3);
  Value probLast = block->getArgument(4);
  Value probData = block->getArgument(5);

  // bundle the inputs into a struct
  Value probRaw = builder.create<circt::hw::StructCreateOp>(
    builder.getUnknownLoc(),
    probStructType,
    std::vector<Value>{probLast, probData}
  ).getResult();

  // wrap probability
  WrapValidReadyOp wrvOp = builder.create<WrapValidReadyOp>(
    builder.getUnknownLoc(),
    probRaw, probValid
  );

  Value probWrapped = wrvOp.getResult(0);
  Value probReady = wrvOp.getResult(1);

  // endpoint
  Value cosimEndpoint = builder.create<CosimEndpointOp>(
    builder.getUnknownLoc(),
    // result type is something like { bits: i40, last: i1 }
    // TODO: generalize bit width for any SPN
    ChannelType::get(builder.getContext(), varStructType),
    clk, rst, probWrapped, builder.getStringAttr("MyEndpoint")      
  ).getResult();

  // unwrap variables
  UnwrapValidReadyOp urvOp = builder.create<UnwrapValidReadyOp>(
    builder.getUnknownLoc(),
    cosimEndpoint, varReady
  );

  Value varRaw = urvOp.getResult(0);
  Value varValid = urvOp.getResult(1);

  // decompose the struct and output it
  SmallVector<Value, 16> outputs{
    varValid
  };

  // varLast
  outputs.push_back(
    builder.create<circt::hw::StructExtractOp>(
      builder.getUnknownLoc(),
      varRaw,
      builder.getStringAttr("last")
    ).getResult()
  );

  // variables
  for (uint32_t i = 0; i < varCount; ++i)
    outputs.push_back(
      builder.create<circt::hw::StructExtractOp>(
        builder.getUnknownLoc(),
        varRaw,
        builder.getStringAttr("v" + std::to_string(i))
      ).getResult()
    );

  outputs.push_back(probReady);

  builder.create<circt::hw::OutputOp>(
    builder.getUnknownLoc(),
    outputs
  );
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
