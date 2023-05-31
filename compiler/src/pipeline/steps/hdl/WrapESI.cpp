#include "WrapESI.hpp"

#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Support/BackedgeBuilder.h"

#include <firp/ESI.hpp>


namespace spnc {

ExecutionResult WrapESI::executeStep(mlir::ModuleOp *root) {
  using namespace circt::hw;
  using namespace circt::esi;

  // assume top name is ReadyValidWrapper

  HWModuleOp hwTop = findTop(*root);

  if (!hwTop)
    return failure("top module not found");

  OpBuilder builder(root->getContext());
  builder.setInsertionPointToEnd(root->getBody());

  FPGAKernel& kernel = getContext()->get<Kernel>()->getFPGAKernel();

  Type enqStructType = lowerType(withLast(vectorType(uintType(kernel.spnBitsPerVar), kernel.spnVarCount)));
  Type deqStructType = lowerType(withLast(uintType(kernel.spnResultWidth)));

  std::vector<circt::hw::PortInfo> wrapperPorts{
    {builder.getStringAttr("clock"), PortDirection::INPUT, builder.getI1Type()},
    {builder.getStringAttr("reset"), PortDirection::INPUT, builder.getI1Type()},
    {builder.getStringAttr("enq"), PortDirection::INPUT, ChannelType::get(builder.getContext(), enqStructType)},
    {builder.getStringAttr("deq"), PortDirection::OUTPUT, ChannelType::get(builder.getContext(), deqStructType)}
  };

  HWModuleOp modOp = builder.create<HWModuleOp>(
    builder.getUnknownLoc(),
    builder.getStringAttr("ESIWrapper"),
    wrapperPorts
  );

  builder.setInsertionPointToStart(modOp.getBodyBlock());

  // unwrap enq
  circt::BackedgeBuilder back(builder, builder.getUnknownLoc());
  circt::Backedge ready = back.get(builder.getI1Type());

  UnwrapValidReadyOp urvOp = builder.create<UnwrapValidReadyOp>(
    builder.getUnknownLoc(),
    builder.getBlock()->getArgument(2), // channel
    ready
  );

  Value rawOutput = urvOp.getResult(0);
  SmallVector<Value> unpacked = firp::esi::unpack(rawOutput, builder);

  circt::Backedge deqReady = back.get(builder.getI1Type());

  std::vector<Value> inputs{
    builder.getBlock()->getArgument(0), // clk
    builder.getBlock()->getArgument(1), // rst
    builder.getBlock()->getArgument(0),  // valid
    // 0, 1, 2, 3, 4
    unpacked[0], unpacked[1], unpacked[2], unpacked[3], unpacked[4],
    // last, ready
    unpacked[5], deqReady
  };

  circt::hw::InstanceOp inst = builder.create<circt::hw::InstanceOp>(
    builder.getUnknownLoc(),
    hwTop,
    builder.getStringAttr(topName + "_inst"),
    inputs
  );

  ready.setValue(inst.getResult(0));

  size_t index = 0;
  Value deqBits = firp::esi::pack(
    SmallVector<Value>{inst.getResult(2), inst.getResult(3)},
    deqStructType,
    builder,
    &index
  );

  // wrap deq
  WrapValidReadyOp rrvOp = builder.create<WrapValidReadyOp>(
    builder.getUnknownLoc(),
    deqBits,
    inst.getResult(1) // deqValid
  );

  deqReady.setValue(rrvOp.getResult(1));

  // TODO: output rrvOp.getResult(0);

  builder.create<OutputOp>(
    builder.getUnknownLoc(),
    rrvOp.getResult(0)
  );

  builder.getBlock()->back().erase();

  root->dump();

  return failure("not implemented");

  /*
  // hw.module @ReadyValidWrapper(%clock: i1, %reset: i1, %enq_valid: i1, %enq_bits_bits_0: i8, %enq_bits_bits_1: i8, %enq_bits_bits_2: i8, %enq_bits_bits_3: i8, %enq_bits_bits_4: i8, %enq_bits_last: i1, %deq_ready: i1)
  // -> (enq_ready: i1, deq_valid: i1, deq_bits_bits: i64, deq_bits_last: i1)

  std::vector<Value> inputs{
    builder.getBlock()->getArgument(0), // clk
    builder.getBlock()->getArgument(1), // rst
    // TODO
  };

  InstanceOp inst = builder.create<InstanceOp>(
    builder.getUnknownLoc(),
    builder.getStringAttr(topName + "_inst"),
    inputs
  );
   */



}

circt::hw::HWModuleOp WrapESI::findTop(mlir::ModuleOp root) {
  using namespace circt::hw;

  HWModuleOp top;

  root.walk([&](HWModuleOp mod) {
    if (mod.getName().str() == topName) {
      if (top)
        top = HWModuleOp();
      else
        top = mod;
    }
  });

  return top;
}

}