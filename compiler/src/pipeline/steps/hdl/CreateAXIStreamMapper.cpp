#include "CreateAXIStreamMapper.hpp"

#include "circt/Dialect/FIRRTL/Passes.h"
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

  // remove firrtl.constCast
  PassManager pm(op.getContext());
  pm.
    nest<circt::firrtl::CircuitOp>().
    nest<circt::firrtl::FModuleOp>().
    addPass(circt::firrtl::createDropConstPass());

  if (mlir::failed(pm.run(op)))
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

  std::string topName = doPrepareForCocoTb ? "AXI4CocoTbTop" : "AXI4StreamMapper";
  setNewTopName(*root, topName);
  CircuitOp circuitOp = cast<CircuitOp>(root->getBody()->front());
  attachFirpContext(circuitOp);
  firpContext()->moduleBuilder->setInitialUid(1000);

  std::string aw = std::to_string(getContext()->get<Kernel>()->getFPGAKernel().memAddrWidth);
  std::string dw = std::to_string(getContext()->get<Kernel>()->getFPGAKernel().memDataWidth);

  FModuleOp loadUnit = insertFIRFile("resources/ipec/IPECLoadUnit_a" + aw + "_d" + dw + ".fir", "IPECLoadUnit");
  FModuleOp storeUnit = insertFIRFile("resources/ipec/IPECStoreUnit_a" + aw + "_d" + dw + ".fir", "IPECStoreUnit");

  if (!loadUnit || !storeUnit)
    return failure("failed to insert IPECLoadUnit_a64_d64.fir or IPECStoreUnit_a64_d64.fir");

  FModuleOp streamWrapper = findModuleByName("AXIStreamWrapper");

  if (!streamWrapper)
    return failure("AXIStreamWrapper module not found");

  const FPGAKernel& kernel = getContext()->get<Kernel>()->getFPGAKernel();

  if (doPrepareForCocoTb) {
    AXI4CocoTbTop tbTop = AXI4CocoTbTop::make(
      kernel,
      loadUnit,
      storeUnit,
      streamWrapper
    );

    tbTop.makeTop();
  } else {
    AXI4StreamMapper mapper = AXI4StreamMapper::make(
      kernel,
      loadUnit,
      storeUnit,
      streamWrapper
    );

    mapper.makeTop();
  }

  if (mlir::failed(firpContext()->finish()))
    return failure("could not verify generated FIRRTL");

  if (mlir::failed(lowerFirrtlToHw()))
    return failure("lowering to HW failed");

  modOp = std::make_unique<mlir::ModuleOp>(firpContext()->root);

  return success();
}

void AXI4StreamMapper::body() {
  auto regs = axi4LiteRegisterFile(
    liteConfig,
    {
      "status",
      "retVal",
      "loadBaseAddress",
      "numLdTransfers",
      "storeBaseAddress",
      "numSdTransfers"
    },
    0x10, // Tapasco uses 0x10 address increments
    io("S_AXI_LITE")
  );

  auto status = regs[0];

  when (status != uval(0), [&](){
    status <<= uval(0);
  });

  auto retVal = regs[1];
  auto loadBaseAddress = regs[2];
  auto numLdTransfers = regs[3];
  auto storeBaseAddress = regs[4];
  auto numSdTransfers = regs[5];

  // instantiate load and store unit
  
  FIRModule loadUnit(ipecLoadUnit);
  loadUnit.io("clock") <<= io("clock");
  loadUnit.io("reset") <<= io("reset");
  loadUnit.io("io")("start") <<= uval(0);
  loadUnit.io("io")("baseAddress") <<= loadBaseAddress;
  loadUnit.io("io")("numTransfers") <<= numLdTransfers;

  //{ ar : { valid : UInt<1>, flip ready : UInt<1>, id : UInt<1>, addr : UInt<32>, len : UInt<8>, size : UInt<3>, burst : UInt<2>, lock : UInt<1>, cache : UInt<4>, prot : UInt<3>, qos : UInt<4>, region : UInt<4>}
  io("M_AXI")("ARVALID") <<= loadUnit.io("io")("axi4master")("ar")("valid");
  io("M_AXI")("ARID") <<= loadUnit.io("io")("axi4master")("ar")("id");
  io("M_AXI")("ARADDR") <<= loadUnit.io("io")("axi4master")("ar")("addr");
  io("M_AXI")("ARLEN") <<= loadUnit.io("io")("axi4master")("ar")("len");
  io("M_AXI")("ARSIZE") <<= loadUnit.io("io")("axi4master")("ar")("size");
  io("M_AXI")("ARBURST") <<= loadUnit.io("io")("axi4master")("ar")("burst");
  io("M_AXI")("ARLOCK") <<= loadUnit.io("io")("axi4master")("ar")("lock");
  io("M_AXI")("ARCACHE") <<= loadUnit.io("io")("axi4master")("ar")("cache");
  io("M_AXI")("ARPROT") <<= loadUnit.io("io")("axi4master")("ar")("prot");
  io("M_AXI")("ARQOS") <<= loadUnit.io("io")("axi4master")("ar")("qos");
  io("M_AXI")("ARREGION") <<= loadUnit.io("io")("axi4master")("ar")("region");
  io("M_AXI")("ARUSER") <<= uval(0, readConfig.userBits);
  loadUnit.io("io")("axi4master")("ar")("ready") <<= io("M_AXI")("ARREADY");

  // flip r : { valid : UInt<1>, flip ready : UInt<1>, id : UInt<1>, data : UInt<32>, resp : UInt<2>, last : UInt<1>}}
  loadUnit.io("io")("axi4master")("r")("valid") <<= io("M_AXI")("RVALID");
  loadUnit.io("io")("axi4master")("r")("id") <<= io("M_AXI")("RID");
  loadUnit.io("io")("axi4master")("r")("data") <<= io("M_AXI")("RDATA");
  loadUnit.io("io")("axi4master")("r")("resp") <<= io("M_AXI")("RRESP");
  loadUnit.io("io")("axi4master")("r")("last") <<= io("M_AXI")("RLAST");
  io("M_AXI")("RREADY") <<= loadUnit.io("io")("axi4master")("r")("ready");

  AXIStreamSender sender(mAxisConfig);
  sender.io("enq") <<= loadUnit.io("io")("data");
  io("M_AXIS") <<= sender.io("AXIS");

  FIRModule storeUnit(ipecStoreUnit);
  storeUnit.io("clock") <<= io("clock");
  storeUnit.io("reset") <<= io("reset");
  storeUnit.io("io")("start") <<= uval(0);
  storeUnit.io("io")("baseAddress") <<= storeBaseAddress;
  storeUnit.io("io")("numTransfers") <<= numSdTransfers;

  // aw : { valid : UInt<1>, flip ready : UInt<1>, id : UInt<1>, addr : UInt<32>, len : UInt<8>, size : UInt<3>, burst : UInt<2>, lock : UInt<1>, cache : UInt<4>, prot : UInt<3>, qos : UInt<4>, region : UInt<4>}
  io("M_AXI")("AWVALID") <<= storeUnit.io("io")("axi4master")("aw")("valid");
  io("M_AXI")("AWID") <<= storeUnit.io("io")("axi4master")("aw")("id");
  io("M_AXI")("AWADDR") <<= storeUnit.io("io")("axi4master")("aw")("addr");
  io("M_AXI")("AWLEN") <<= storeUnit.io("io")("axi4master")("aw")("len");
  io("M_AXI")("AWSIZE") <<= storeUnit.io("io")("axi4master")("aw")("size");
  io("M_AXI")("AWBURST") <<= storeUnit.io("io")("axi4master")("aw")("burst");
  io("M_AXI")("AWLOCK") <<= storeUnit.io("io")("axi4master")("aw")("lock");
  io("M_AXI")("AWCACHE") <<= storeUnit.io("io")("axi4master")("aw")("cache");
  io("M_AXI")("AWPROT") <<= storeUnit.io("io")("axi4master")("aw")("prot");
  io("M_AXI")("AWQOS") <<= storeUnit.io("io")("axi4master")("aw")("qos");
  io("M_AXI")("AWREGION") <<= storeUnit.io("io")("axi4master")("aw")("region");
  io("M_AXI")("AWUSER") <<= uval(0, writeConfig.userBits);
  storeUnit.io("io")("axi4master")("aw")("ready") <<= io("M_AXI")("AWREADY");

  // w : { valid : UInt<1>, flip ready : UInt<1>, data : UInt<32>, strb : UInt<4>, last : UInt<1>}
  io("M_AXI")("WVALID") <<= storeUnit.io("io")("axi4master")("w")("valid");
  io("M_AXI")("WDATA") <<= storeUnit.io("io")("axi4master")("w")("data");
  io("M_AXI")("WSTRB") <<= storeUnit.io("io")("axi4master")("w")("strb");
  io("M_AXI")("WLAST") <<= storeUnit.io("io")("axi4master")("w")("last");
  storeUnit.io("io")("axi4master")("w")("ready") <<= io("M_AXI")("WREADY");

  // flip b : { valid : UInt<1>, flip ready : UInt<1>, id : UInt<1>, resp : UInt<2>}}
  storeUnit.io("io")("axi4master")("b")("valid") <<= io("M_AXI")("BVALID");
  storeUnit.io("io")("axi4master")("b")("id") <<= io("M_AXI")("BID");
  storeUnit.io("io")("axi4master")("b")("resp") <<= io("M_AXI")("BRESP");
  io("M_AXI")("BREADY") <<= storeUnit.io("io")("axi4master")("b")("ready");

  AXIStreamReceiver receiver(sAxisConfig);
  receiver.io("AXIS") <<= io("S_AXIS");
  storeUnit.io("io")("data") <<= receiver.io("deq");

  // connect controller directly to the outside world
  FIRModule spnController(spnAxisController);
  spnController.io("clock") <<= io("clock");
  spnController.io("reset") <<= io("reset");
  spnController.io("AXIS_SLAVE") <<= io("S_AXIS_CONTROLLER");
  io("M_AXIS_CONTROLLER") <<= spnController.io("AXIS_MASTER");

  assert(
    io("M_AXIS_CONTROLLER")("TDATA").bitCount() == spnController.io("AXIS_MASTER")("TDATA").bitCount()
  );

  // FSM
  enum State {
    IDLE,
    RUNNING,
    DONE
  };

  auto state = regInit(uval(0, 2), "state");
  auto cycleCount = regInit(uval(0, 32), "cycleCount");
  auto maxCycleCount = regInit(uval(100000, 32), "maxCycleCount");

  io("interrupt") <<= uval(0);

  when (state.read() == uval(IDLE), [&](){
    when (status(0), [&](){
      state <<= uval(RUNNING);
      loadUnit.io("io")("start") <<= uval(1);
      storeUnit.io("io")("start") <<= uval(1);
      cycleCount <<= uval(0);
    });
  })
  .elseWhen (state.read() == uval(RUNNING), [&](){
    when (loadUnit.io("io")("done") & storeUnit.io("io")("done"), [&](){
      state <<= uval(DONE);
      io("interrupt") <<= uval(1);
      retVal <<= cycleCount;
    })
    .elseWhen (cycleCount.read() >= maxCycleCount.read(), [&](){
      state <<= uval(DONE);
      io("interrupt") <<= uval(1);
      retVal <<= uval(0);
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
    .addrBits = uint32_t(kernel.liteAddrWidth),
    .dataBits = uint32_t(kernel.liteDataWidth)
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

void AXI4CocoTbTop::body() {
  AXI4StreamMapper mapper(
    liteConfig,
    writeConfig,
    readConfig,
    mAxisConfig,
    sAxisControllerConfig,
    sAxisConfig,
    mAxisControllerConfig,
    ipecLoadUnit,
    ipecStoreUnit,
    spnAxisController
  );

  io("M_AXI") <<= mapper.io("M_AXI");
  mapper.io("S_AXI_LITE") <<= io("S_AXI_LITE");

  AXIStreamConverter converter1(mAxisConfig, sAxisControllerConfig);
  converter1.io("AXIS_slave") <<= mapper.io("M_AXIS");
  mapper.io("S_AXIS_CONTROLLER") <<= converter1.io("AXIS_master");

  AXIStreamConverter converter2(mAxisControllerConfig, sAxisConfig);
  converter2.io("AXIS_slave") <<= mapper.io("M_AXIS_CONTROLLER");
  mapper.io("S_AXIS") <<= converter2.io("AXIS_master");

  mapper.io("S_AXI_LITE") <<= io("S_AXI_LITE");
  io("M_AXI") <<= mapper.io("M_AXI");

  io("interrupt") <<= mapper.io("interrupt");

  svCocoTBVerbatim("AXI4CocoTbTop");
}

AXI4CocoTbTop AXI4CocoTbTop::make(
  const FPGAKernel& kernel,
  circt::firrtl::FModuleOp ipecLoadUnit,
  circt::firrtl::FModuleOp ipecStoreUnit,
  circt::firrtl::FModuleOp spnAxisController
) {
  // I think TaPaSco uses this
  axi4lite::AXI4LiteConfig liteConfig{
    .addrBits = uint32_t(kernel.liteAddrWidth),
    .dataBits = uint32_t(kernel.liteDataWidth)
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

  return AXI4CocoTbTop(
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

void DummyWrapper::body() {
  auto regs = axi4LiteRegisterFile(
    liteConfig,
    {
      "status",
      "retVal",
      "a",
      "b",
      "c",
      "d"
    },
    0x10, // Tapasco uses 0x10 address increments
    io("S_AXI_LITE")
  );

  /*
  io("S_AXI_LITE")("ARREADY") <<= uval(0);
  
  io("S_AXI_LITE")("RVALID") <<= uval(0);
  io("S_AXI_LITE")("RRESP") <<= uval(0);
  io("S_AXI_LITE")("RDATA") <<= uval(0);
  
  io("S_AXI_LITE")("AWREADY") <<= uval(0);
  
  io("S_AXI_LITE")("WREADY") <<= uval(0);

  io("S_AXI_LITE")("BVALID") <<= uval(0);
  io("S_AXI_LITE")("BRESP") <<= uval(0);
   */

  auto status = regs[0];

  when (status != uval(0), [&](){
    status <<= uval(0);
  });

  auto retVal = regs[1];
  auto a = regs[2];
  auto b = regs[3];
  auto c = regs[4];
  auto d = regs[5];

  //{ ar : { valid : UInt<1>, flip ready : UInt<1>, id : UInt<1>, addr : UInt<32>, len : UInt<8>, size : UInt<3>, burst : UInt<2>, lock : UInt<1>, cache : UInt<4>, prot : UInt<3>, qos : UInt<4>, region : UInt<4>}
  io("M_AXI")("ARVALID") <<= uval(0);
  io("M_AXI")("ARID") <<= uval(0);
  io("M_AXI")("ARADDR") <<= uval(0);
  io("M_AXI")("ARLEN") <<= uval(0);
  io("M_AXI")("ARSIZE") <<= uval(0);
  io("M_AXI")("ARBURST") <<= uval(0);
  io("M_AXI")("ARLOCK") <<= uval(0);
  io("M_AXI")("ARCACHE") <<= uval(0);
  io("M_AXI")("ARPROT") <<= uval(0);
  io("M_AXI")("ARQOS") <<= uval(0);
  io("M_AXI")("ARREGION") <<= uval(0);
  io("M_AXI")("ARUSER") <<= uval(0, readConfig.userBits);

  // flip r : { valid : UInt<1>, flip ready : UInt<1>, id : UInt<1>, data : UInt<32>, resp : UInt<2>, last : UInt<1>}}
  io("M_AXI")("RREADY") <<= uval(0);

  // aw : { valid : UInt<1>, flip ready : UInt<1>, id : UInt<1>, addr : UInt<32>, len : UInt<8>, size : UInt<3>, burst : UInt<2>, lock : UInt<1>, cache : UInt<4>, prot : UInt<3>, qos : UInt<4>, region : UInt<4>}
  io("M_AXI")("AWVALID") <<= uval(0);
  io("M_AXI")("AWID") <<= uval(0);
  io("M_AXI")("AWADDR") <<= uval(0);
  io("M_AXI")("AWLEN") <<= uval(0);
  io("M_AXI")("AWSIZE") <<= uval(0);
  io("M_AXI")("AWBURST") <<= uval(0);
  io("M_AXI")("AWLOCK") <<= uval(0);
  io("M_AXI")("AWCACHE") <<= uval(0);
  io("M_AXI")("AWPROT") <<= uval(0);
  io("M_AXI")("AWQOS") <<= uval(0);
  io("M_AXI")("AWREGION") <<= uval(0);
  io("M_AXI")("AWUSER") <<= uval(0, writeConfig.userBits);

  // w : { valid : UInt<1>, flip ready : UInt<1>, data : UInt<32>, strb : UInt<4>, last : UInt<1>}
  io("M_AXI")("WVALID") <<= uval(0);
  io("M_AXI")("WDATA") <<= uval(0);
  io("M_AXI")("WSTRB") <<= uval(0);
  io("M_AXI")("WLAST") <<= uval(0);

  // flip b : { valid : UInt<1>, flip ready : UInt<1>, id : UInt<1>, resp : UInt<2>}}
  io("M_AXI")("BREADY") <<= uval(0);

  // FSM
  enum State {
    IDLE = 0,
    RUNNING,
    DONE
  };

  auto state = regInit(uval(0, 2), "state");
  auto interrupt = regInit(uval(1), "interrupt");

  io("interrupt") <<= interrupt;
  retVal <<= uval(0);

  when (state.read() == uval(IDLE), [&](){
    when (status(0), [&](){
      state <<= uval(RUNNING);
    });
  })
  .elseWhen (state.read() == uval(RUNNING), [&](){
    interrupt <<= uval(1);
    state <<= uval(DONE);
  })
  .elseWhen (state.read() == uval(DONE), [&](){
    interrupt <<= uval(0);
    state <<= uval(IDLE);
  });
}

DummyWrapper DummyWrapper::make(const FPGAKernel& kernel) {
  // I think TaPaSco uses this
  axi4lite::AXI4LiteConfig liteConfig{
    .addrBits = uint32_t(kernel.liteAddrWidth),
    .dataBits = uint32_t(kernel.liteDataWidth)
  };

  // used for the write and read channels of the memory AXI4 ports
  axi4::AXI4Config memConfig{
    .addrBits = uint32_t(kernel.memAddrWidth),
    .dataBits = uint32_t(kernel.memDataWidth)
  };

  return DummyWrapper(
    liteConfig,
    memConfig,
    memConfig
  );
}

void RegisterFile::body() {
  auto regs = axi4LiteRegisterFile(
    liteConfig,
    { "status", "retVal", "a", "b", "c" },
    0x10,
    io("S_AXI_LITE")
  );

  auto status = regs[0];

  when (status != uval(0), [&](){
    status <<= uval(0);
  });

  auto retVal = regs[1];
  auto a = regs[2];
  auto b = regs[3];
  auto c = regs[4];

  //{ ar : { valid : UInt<1>, flip ready : UInt<1>, id : UInt<1>, addr : UInt<32>, len : UInt<8>, size : UInt<3>, burst : UInt<2>, lock : UInt<1>, cache : UInt<4>, prot : UInt<3>, qos : UInt<4>, region : UInt<4>}
  io("M_AXI")("ARVALID") <<= uval(0);
  io("M_AXI")("ARID") <<= uval(0);
  io("M_AXI")("ARADDR") <<= uval(0);
  io("M_AXI")("ARLEN") <<= uval(0);
  io("M_AXI")("ARSIZE") <<= uval(0);
  io("M_AXI")("ARBURST") <<= uval(0);
  io("M_AXI")("ARLOCK") <<= uval(0);
  io("M_AXI")("ARCACHE") <<= uval(0);
  io("M_AXI")("ARPROT") <<= uval(0);
  io("M_AXI")("ARQOS") <<= uval(0);
  io("M_AXI")("ARREGION") <<= uval(0);
  io("M_AXI")("ARUSER") <<= uval(0, readConfig.userBits);

  // flip r : { valid : UInt<1>, flip ready : UInt<1>, id : UInt<1>, data : UInt<32>, resp : UInt<2>, last : UInt<1>}}
  io("M_AXI")("RREADY") <<= uval(0);

  // aw : { valid : UInt<1>, flip ready : UInt<1>, id : UInt<1>, addr : UInt<32>, len : UInt<8>, size : UInt<3>, burst : UInt<2>, lock : UInt<1>, cache : UInt<4>, prot : UInt<3>, qos : UInt<4>, region : UInt<4>}
  io("M_AXI")("AWVALID") <<= uval(0);
  io("M_AXI")("AWID") <<= uval(0);
  io("M_AXI")("AWADDR") <<= uval(0);
  io("M_AXI")("AWLEN") <<= uval(0);
  io("M_AXI")("AWSIZE") <<= uval(0);
  io("M_AXI")("AWBURST") <<= uval(0);
  io("M_AXI")("AWLOCK") <<= uval(0);
  io("M_AXI")("AWCACHE") <<= uval(0);
  io("M_AXI")("AWPROT") <<= uval(0);
  io("M_AXI")("AWQOS") <<= uval(0);
  io("M_AXI")("AWREGION") <<= uval(0);
  io("M_AXI")("AWUSER") <<= uval(0, writeConfig.userBits);

  // w : { valid : UInt<1>, flip ready : UInt<1>, data : UInt<32>, strb : UInt<4>, last : UInt<1>}
  io("M_AXI")("WVALID") <<= uval(0);
  io("M_AXI")("WDATA") <<= uval(0);
  io("M_AXI")("WSTRB") <<= uval(0);
  io("M_AXI")("WLAST") <<= uval(0);

  // flip b : { valid : UInt<1>, flip ready : UInt<1>, id : UInt<1>, resp : UInt<2>}}
  io("M_AXI")("BREADY") <<= uval(0);

  // FSM
  enum State {
    IDLE = 0,
    RUNNING,
    DONE
  };

  auto state = regInit(uval(0, 2), "state");
  auto interrupt = regInit(uval(0), "interrupt");
  auto counter = regInit(uval(0, 32), "counter");

  io("interrupt") <<= interrupt;
  retVal <<= uval(0);

  when (state.read() == uval(IDLE), [&](){
    when (status(0), [&](){
      state <<= uval(RUNNING);
      counter <<= a;
      c <<= uval(0);
    });
  })
  .elseWhen (state.read() == uval(RUNNING), [&](){
    counter <<= counter.read() - uval(1);
    c <<= c + b;

    when (counter.read() == uval(0), [&](){
      interrupt <<= uval(1);
      state <<= uval(DONE);
    });
  })
  .elseWhen (state.read() == uval(DONE), [&](){
    interrupt <<= uval(0);
    state <<= uval(IDLE);
  });

  svCocoTBVerbatim("RegisterFile");
}

}