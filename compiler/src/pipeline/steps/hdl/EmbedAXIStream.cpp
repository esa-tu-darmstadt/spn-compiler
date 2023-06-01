#include "EmbedAXIStream.hpp"

#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"

#include <firp/lowering.hpp>


namespace spnc {

static FModuleOp findSPNBody(mlir::ModuleOp *circuitRoot) {
  FModuleOp spnBody;

  circuitRoot->walk([&](FModuleOp op){
    if (op.getName() == "SPNBody") {
      if (spnBody)
        assert(false && "multiple SPNBody modules found");

      spnBody = op;
    }
  });

  return spnBody;
}

ExecutionResult EmbedReadyValid::executeStep(mlir::ModuleOp *circuitRoot) {
  FModuleOp spnBody = findSPNBody(circuitRoot);

  if (!spnBody)
    return failure("SPNBody module not found");

  std::string topName = "ReadyValidWrapper";
  if (failed(setNewTopName(*circuitRoot, topName)))
      return failure("failed to rename circuit op");

  CircuitOp circuitOp = dyn_cast<CircuitOp>(circuitRoot->getBody()->front());

  FPGAKernel& kernel = getContext()->get<Kernel>()->getFPGAKernel();
  kernel.bodyDelay = dyn_cast<IntegerAttr>(spnBody->getAttr("fpga.body_delay")).getInt();
  kernel.fifoDepth = kernel.bodyDelay * 2;

  // create the wrapper
  initFirpContext(circuitOp);

  ReadyValidWrapper wrapper(
    spnBody,
    kernel.spnVarCount, kernel.spnBitsPerVar, kernel.spnResultWidth,
    kernel.fifoDepth, kernel.bodyDelay
  );
  wrapper.makeTop();

  firpContext()->finish();

  if (mlir::failed(lowerFirrtlToHw()))
    return failure("lowering to HW failed");

  topModule = std::make_unique<ModuleOp>(firpContext()->root);

  return success();
}

ExecutionResult EmbedAXIStream::executeStep(mlir::ModuleOp *circuitRoot) {
  FModuleOp spnBody = findSPNBody(circuitRoot);

  if (!spnBody)
    return failure("SPNBody module not found");

  std::string topName = "AXIStreamWrapper";
  if (failed(setNewTopName(*circuitRoot, topName)))
      return failure("failed to rename circuit op");

  CircuitOp circuitOp = dyn_cast<CircuitOp>(circuitRoot->getBody()->front());

  FPGAKernel& kernel = getContext()->get<Kernel>()->getFPGAKernel();
  kernel.bodyDelay = dyn_cast<IntegerAttr>(spnBody->getAttr("fpga.body_delay")).getInt();
  kernel.fifoDepth = kernel.bodyDelay * 2;

  // create the wrapper
  initFirpContext(circuitOp);

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

  AXIStreamWrapper wrapper(
    spnBody,
    slaveConfig, masterConfig,
    kernel.spnVarCount, kernel.spnBitsPerVar, kernel.spnResultWidth,
    kernel.fifoDepth, kernel.bodyDelay
  );

  wrapper.makeTop();

  firpContext()->finish();
  firpContext()->dump();

  if (mlir::failed(lowerFirrtlToHw()))
    return failure("lowering to HW failed");

  topModule = std::make_unique<ModuleOp>(firpContext()->root);

  return success();
}

void ReadyValidWrapper::body() {
  auto enqFire = wireInit(doesFire(io("enq")), "enqFire");

  FirpQueue fifo(withLast(uintType(resultWidth)), fifoDepth);
  io("deq") <<= fifo.io("deq");
  auto deqFire = wireInit(doesFire(fifo.io("enq")), "deqFire");

  // a bit of manual work because FIRRTL++ wasn't designed for this
  InstanceOp instOp = firpContext()->builder().create<InstanceOp>(
    firpContext()->builder().getUnknownLoc(),
    spnBody,
    firpContext()->builder().getStringAttr("SPNBody_instance")
  );

  FValue(instOp.getResult(0)) <<= io(firpContext()->getDefaultClockName());
  FValue(instOp.getResult(1)) <<= io(firpContext()->getDefaultResetName());
  FValue spnBodyOut = instOp.getResult(2);

  for (uint32_t i = 0; i < spnVarCount; ++i)
    FValue(instOp.getResult(i + 3)) <<= io("enq")("bits")("bits")[i];

  ShiftRegister lastDelayed(bitType(), bodyDelay);
  lastDelayed.io("in") <<= io("enq")("bits")("last");

  ShiftRegister validDelayed(bitType(), bodyDelay);
  validDelayed.io("in") <<= io("enq")("valid");

  fifo.io("enq")("bits")("bits") <<= spnBodyOut;
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

void AXIStreamWrapper::body() {
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

  // a bit of manual work because FIRRTL++ wasn't designed for this
  InstanceOp instOp = firpContext()->builder().create<InstanceOp>(
    firpContext()->builder().getUnknownLoc(),
    spnBody,
    firpContext()->builder().getStringAttr("SPNBody_instance")
  );

  FValue(instOp.getResult(0)) <<= io(firpContext()->getDefaultClockName());
  FValue(instOp.getResult(1)) <<= io(firpContext()->getDefaultResetName());
  FValue spnBodyOut = instOp.getResult(2);

  for (uint32_t i = 0; i < spnVarCount; ++i)
    FValue(instOp.getResult(i + 3)) <<= receiver.io("deq")("bits")("bits")((i + 1) * bitsPerVar - 1, i * bitsPerVar);

  // TODO: Build controller after scheduling!
  //size_t bodyDelay = fifoDepth;
  ShiftRegister lastDelayed(bitType(), bodyDelay);
  lastDelayed.io("in") <<= receiver.io("deq")("bits")("last");

  ShiftRegister validDelayed(bitType(), bodyDelay);
  validDelayed.io("in") <<= receiver.io("deq")("valid");

  fifo.io("enq")("bits")("bits") <<= spnBodyOut;
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

}