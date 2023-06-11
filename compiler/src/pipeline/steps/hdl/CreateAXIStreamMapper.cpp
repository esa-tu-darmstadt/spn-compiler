#include "CreateAXIStreamMapper.hpp"

#include <firp/lowering.hpp>


namespace spnc {

using namespace firp;
using namespace firp::axis;

FModuleOp CreateAXIStreamMapper::findModuleByName(const std::string& name) {
  FModuleOp mod;

  firpContext()->circuitOp.walk([&](FModuleOp op){
    if (op.getName() == name) {
      if (mod)
        assert(false && "multiple modules found");

      mod = op;
    }
  });

  return mod;
}

FModuleOp CreateAXIStreamMapper::insertFIRFile(const std::filesystem::path& path, const std::string& moduleName) {
  ModuleOp op = importFIRFile(path);

  if (!op)
    return FModuleOp();

  auto bodyBlock = cast<CircuitOp>(op.getBody()->front()) // CircuitOp
    .getBodyBlock(); //->begin(); // iterator on first FModuleOp

  //auto targetBlock = firpContext()->circuitOp.getBodyBlock();

  OpBuilder builder(firpContext()->context());
  builder.setInsertionPointToEnd(firpContext()->circuitOp.getBodyBlock());

  for (Operation &op : *bodyBlock) {
    Operation *clone = op.clone();
    builder.insert(clone);
  }

  return findModuleByName(moduleName);
}

ExecutionResult CreateAXIStreamMapper::executeStep(mlir::ModuleOp *root) {
  llvm::outs() << "CreateAXIStreamMapper::executeStep()\n";

  setNewTopName(*root, "AXI4StreamMapper");
  CircuitOp circuitOp = cast<CircuitOp>(root->getBody()->front());
  attachFirpContext(circuitOp);
  firpContext()->moduleBuilder->setInitialUid(1000);

  FModuleOp loadUnit = insertFIRFile("resources/ipec/IPECLoadUnit_a32_d32.fir", "IPECLoadUnit");
  FModuleOp storeUnit = insertFIRFile("resources/ipec/IPECStoreUnit_a32_d32.fir", "IPECStoreUnit");

  if (!loadUnit || !storeUnit)
    return failure("failed to insert IPECLoadUnit_a64_d64.fir or IPECStoreUnit_a64_d64.fir");

  FModuleOp streamWrapper = findModuleByName("AXIStreamWrapper");

  if (!streamWrapper)
    return failure("AXIStreamWrapper module not found");

  const FPGAKernel& kernel = getContext()->get<Kernel>()->getFPGAKernel();

  AXI4StreamMapper mapper = AXI4StreamMapper::make(
    kernel,
    loadUnit,
    storeUnit,
    streamWrapper
  );

  mapper.makeTop();
  
  firpContext()->dump();
  
  assert(mlir::succeeded(firpContext()->finish()));
  
  assert(false && "fuck me");

  firpContext()->finish();
  firpContext()->dump();

  assert(false && "not implemented");
  return failure("not implemented");
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

  return;

  // instantiate load and store unit
  
  FIRModule loadUnit(ipecLoadUnit);
  loadUnit.io("clock") <<= io("clock");
  loadUnit.io("reset") <<= io("reset");
  loadUnit.io("io")("start") <<= status(0);
  loadUnit.io("io")("baseAddress") <<= loadBaseAddress;
  loadUnit.io("io")("numTransfers") <<= numLdTransfers;

  AXIStreamSender sender(mAxisConfig);
  sender.io("enq") <<= loadUnit.io("io")("data");
  io("M_AXIS") <<= sender.io("AXIS");

  FIRModule storeUnit(ipecStoreUnit);
  storeUnit.io("clock") <<= io("clock");
  storeUnit.io("reset") <<= io("reset");
  storeUnit.io("io")("start") <<= status(0);
  storeUnit.io("io")("baseAddress") <<= storeBaseAddress;
  storeUnit.io("io")("numTransfers") <<= numSdTransfers;

  AXIStreamReceiver receiver(sAxisConfig);
  receiver.io("AXIS") <<= io("S_AXIS");
  storeUnit.io("io")("data") <<= receiver.io("deq");

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
    when (loadUnit.io("io")("done") & storeUnit.io("io")("done"), [&](){
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