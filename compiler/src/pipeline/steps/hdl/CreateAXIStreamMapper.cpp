#include "CreateAXIStreamMapper.hpp"


namespace spnc {

using namespace firp;
using namespace firp::axis;

ExecutionResult CreateAXIStreamMapper::executeStep(mlir::ModuleOp *root) {
  



  return failure();
}

void AXI4StreamMapper::body() {
  axi4lite::AXI4LiteRegisterFile regs(
    liteConfig,
    {
      "status",
      "retVal",
      "loadBaseAddress",
      "numLdTransfers",
      "storeBaseAddress",
      "numSdTransfers"
    }
  );

  auto status = regs.io("status");
  auto retVal = regs.io("retVal");
  auto loadBaseAddress = regs.io("loadBaseAddress");
  auto numLdTransfers = regs.io("numLdTransfers");
  auto storeBaseAddress = regs.io("storeBaseAddress");
  auto numSdTransfers = regs.io("numSdTransfers");

  regs.io("AXI4LiteSlave") <<= io("S_AXI_LITE");

  // instantiate load and store unit
  
  FIRModule loadUnit(ipecLoadUnit);
  loadUnit.io("start") <<= status(0);
  loadUnit.io("baseAddress") <<= loadBaseAddress;
  loadUnit.io("numTransfers") <<= numLdTransfers;

  AXIStreamSender sender(mAxisConfig);
  sender.io("enq") <<= loadUnit.io("data");
  io("M_AXIS") <<= sender.io("AXIS");

  FIRModule storeUnit(ipecStoreUnit);
  storeUnit.io("start") <<= status(0);
  storeUnit.io("baseAddress") <<= storeBaseAddress;
  storeUnit.io("numTransfers") <<= numSdTransfers;

  AXIStreamReceiver receiver(sAxisConfig);
  receiver.io("AXIS") <<= io("S_AXIS");
  storeUnit.io("data") <<= receiver.io("deq");

  // connect controller directly to the outside world
  FIRModule spnController(spnAxisController);
  spnController.io("AXIS_SLAVE") <<= io("S_AXIS_CONTROLLER");
  io("M_AXIS_CONTROLLER") <<= spnController.io("AXIS_MASTER");

  // FSM
  enum State {
    IDLE,
    RUNNING,
    DONE
  };

  auto state = regInit(uval(0, 2), "state");
  auto cycleCount = regInit(uval(0, 32), "cycleCount");

  when (state.read() == uval(IDLE), [&](){
    when (status == uval(0), [&](){
      state <<= uval(RUNNING);
    });
  })
  .elseWhen (state.read() == uval(RUNNING), [&](){
    when (loadUnit.io("done") & storeUnit.io("done"), [&](){
      state <<= uval(DONE);
      io("interrupt") <<= uval(1);
      retVal <<= cycleCount;
    });

    cycleCount <<= cycleCount.read() + uval(1);
  })
  .elseWhen (state.read() == uval(DONE), [&](){
    io("interrupt") <<= uval(0);
    state <<= uval(IDLE);
  });
}

AXI4StreamMapper AXI4StreamMapper::make(
    const FPGAKernel& kernel,
    circt::firrtl::FModuleOp ipecLoadUnit,
    circt::firrtl::FModuleOp ipecStoreUnit,
    circt::firrtl::FModuleOp spnAxisController
) {
  // I think TaPaSco uses this
  axi4lite::AXI4LiteConfig liteConfig{
    .addrBits = 64,
    .dataBits = 128
  };

  // used for the write and read channels of the memory AXI4 ports
  axi4::AXI4Config memConfig{
    .addrBits = uint32_t(kernel.memAddrWidth),
    .dataBits = uint32_t(kernel.memDataWidth)
  };

  // is connected to the memory AXI4 port
  firp::axis::AXIStreamConfig mAxisConfig{
    .dataBits = uint32_t(kernel.memDataWidth)
  };

  // is connected to the memory AXI4 port
  firp::axis::AXIStreamConfig sAxisConfig{
    .dataBits = uint32_t(kernel.memDataWidth)
  };

  // is connected to the controller input
  firp::axis::AXIStreamConfig sAxisControllerConfig{
    .dataBits = uint32_t(kernel.sAxisControllerWidth)
  };

  // is connected to the controller output
  firp::axis::AXIStreamConfig mAxisControllerConfig{
    .dataBits = uint32_t(kernel.mAxisControllerWidth)
  };

  return AXI4StreamMapper(
    liteConfig,
    memConfig,
    memConfig,
    mAxisConfig,
    sAxisControllerConfig,
    sAxisConfig,
    mAxisControllerConfig,
    ipecLoadUnit,
    ipecStoreUnit,
    spnAxisController
  );
}

}