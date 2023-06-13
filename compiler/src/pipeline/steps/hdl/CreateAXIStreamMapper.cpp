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

  FModuleOp loadUnit = insertFIRFile("resources/ipec/IPECLoadUnit_a32_d32.fir", "IPECLoadUnit");
  FModuleOp storeUnit = insertFIRFile("resources/ipec/IPECStoreUnit_a32_d32.fir", "IPECStoreUnit");

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

  firpContext()->dump();

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
    io("S_AXI_LITE")
  );

  auto status = regs[0];
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
  io("M_AXI")("AR")("VALID") <<= loadUnit.io("io")("axi4master")("ar")("valid");
  io("M_AXI")("AR")("ID") <<= loadUnit.io("io")("axi4master")("ar")("id");
  io("M_AXI")("AR")("ADDR") <<= loadUnit.io("io")("axi4master")("ar")("addr");
  io("M_AXI")("AR")("LEN") <<= loadUnit.io("io")("axi4master")("ar")("len");
  io("M_AXI")("AR")("SIZE") <<= loadUnit.io("io")("axi4master")("ar")("size");
  io("M_AXI")("AR")("BURST") <<= loadUnit.io("io")("axi4master")("ar")("burst");
  io("M_AXI")("AR")("LOCK") <<= loadUnit.io("io")("axi4master")("ar")("lock");
  io("M_AXI")("AR")("CACHE") <<= loadUnit.io("io")("axi4master")("ar")("cache");
  io("M_AXI")("AR")("PROT") <<= loadUnit.io("io")("axi4master")("ar")("prot");
  io("M_AXI")("AR")("QOS") <<= loadUnit.io("io")("axi4master")("ar")("qos");
  io("M_AXI")("AR")("REGION") <<= loadUnit.io("io")("axi4master")("ar")("region");
  io("M_AXI")("AR")("USER") <<= uval(0, readConfig.userBits);
  loadUnit.io("io")("axi4master")("ar")("ready") <<= io("M_AXI")("AR")("READY");

  // flip r : { valid : UInt<1>, flip ready : UInt<1>, id : UInt<1>, data : UInt<32>, resp : UInt<2>, last : UInt<1>}}
  loadUnit.io("io")("axi4master")("r")("valid") <<= io("M_AXI")("R")("VALID");
  loadUnit.io("io")("axi4master")("r")("id") <<= io("M_AXI")("R")("ID");
  loadUnit.io("io")("axi4master")("r")("data") <<= io("M_AXI")("R")("DATA");
  loadUnit.io("io")("axi4master")("r")("resp") <<= io("M_AXI")("R")("RESP");
  loadUnit.io("io")("axi4master")("r")("last") <<= io("M_AXI")("R")("LAST");
  io("M_AXI")("R")("READY") <<= loadUnit.io("io")("axi4master")("r")("ready");

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
  io("M_AXI")("AW")("VALID") <<= storeUnit.io("io")("axi4master")("aw")("valid");
  io("M_AXI")("AW")("ID") <<= storeUnit.io("io")("axi4master")("aw")("id");
  io("M_AXI")("AW")("ADDR") <<= storeUnit.io("io")("axi4master")("aw")("addr");
  io("M_AXI")("AW")("LEN") <<= storeUnit.io("io")("axi4master")("aw")("len");
  io("M_AXI")("AW")("SIZE") <<= storeUnit.io("io")("axi4master")("aw")("size");
  io("M_AXI")("AW")("BURST") <<= storeUnit.io("io")("axi4master")("aw")("burst");
  io("M_AXI")("AW")("LOCK") <<= storeUnit.io("io")("axi4master")("aw")("lock");
  io("M_AXI")("AW")("CACHE") <<= storeUnit.io("io")("axi4master")("aw")("cache");
  io("M_AXI")("AW")("PROT") <<= storeUnit.io("io")("axi4master")("aw")("prot");
  io("M_AXI")("AW")("QOS") <<= storeUnit.io("io")("axi4master")("aw")("qos");
  io("M_AXI")("AW")("REGION") <<= storeUnit.io("io")("axi4master")("aw")("region");
  io("M_AXI")("AW")("USER") <<= uval(0, writeConfig.userBits);
  storeUnit.io("io")("axi4master")("aw")("ready") <<= io("M_AXI")("AW")("READY");

  // w : { valid : UInt<1>, flip ready : UInt<1>, data : UInt<32>, strb : UInt<4>, last : UInt<1>}
  io("M_AXI")("W")("VALID") <<= storeUnit.io("io")("axi4master")("w")("valid");
  io("M_AXI")("W")("DATA") <<= storeUnit.io("io")("axi4master")("w")("data");
  io("M_AXI")("W")("STRB") <<= storeUnit.io("io")("axi4master")("w")("strb");
  io("M_AXI")("W")("LAST") <<= storeUnit.io("io")("axi4master")("w")("last");
  storeUnit.io("io")("axi4master")("w")("ready") <<= io("M_AXI")("W")("READY");

  // flip b : { valid : UInt<1>, flip ready : UInt<1>, id : UInt<1>, resp : UInt<2>}}
  storeUnit.io("io")("axi4master")("b")("valid") <<= io("M_AXI")("B")("VALID");
  storeUnit.io("io")("axi4master")("b")("id") <<= io("M_AXI")("B")("ID");
  storeUnit.io("io")("axi4master")("b")("resp") <<= io("M_AXI")("B")("RESP");
  io("M_AXI")("B")("READY") <<= storeUnit.io("io")("axi4master")("b")("ready");

  AXIStreamReceiver receiver(sAxisConfig);
  receiver.io("AXIS") <<= io("S_AXIS");
  storeUnit.io("io")("data") <<= receiver.io("deq");

  // connect controller directly to the outside world
  FIRModule spnController(spnAxisController);
  spnController.io("clock") <<= io("clock");
  spnController.io("reset") <<= io("reset");
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

  io("interrupt") <<= uval(0);

  when (state.read() == uval(IDLE), [&](){
    when (status(0), [&](){
      state <<= uval(RUNNING);
      loadUnit.io("io")("start") <<= uval(1);
      storeUnit.io("io")("start") <<= uval(1);
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
    .addrBits = 32,
    .dataBits = 32
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
    .addrBits = 32,
    .dataBits = 32
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

}