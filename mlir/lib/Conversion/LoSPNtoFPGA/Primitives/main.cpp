#include <iostream>
#include "LoSPNtoFPGA/Primitives/Primitives.hpp"


int main(int argc, const char **argv)
{
  std::unique_ptr<mlir::MLIRContext> context = std::make_unique<mlir::MLIRContext>();

  assert(context->getOrLoadDialect<circt::hw::HWDialect>());
  assert(context->getOrLoadDialect<circt::seq::SeqDialect>());
  assert(context->getOrLoadDialect<circt::firrtl::FIRRTLDialect>());

  using namespace ::mlir::spn::fpga::primitives;
  using namespace ::mlir::spn::fpga::primitives::operators;
  using namespace ::circt::firrtl;
  using namespace ::mlir;

  Value clk;
  initPrimitiveBuilder(context.get(), clk);

  ModuleOp root = getPrimitiveBuilder()->builder.create<ModuleOp>(
    getPrimitiveBuilder()->builder.getUnknownLoc()
  );

  getPrimitiveBuilder()->builder.setInsertionPointToStart(
    &root.getBodyRegion().front()
  );

  auto valid = UInt(1, 1);
  auto ready = UInt(1, 1);
  auto count = UInt(123, 16);

  Value canEnqueue = (valid & ready & (count < lift(constant(8, 16))))->build();

  Value lower = bits(count, 3, 0)->build();

  auto reg = Reg(UInt(32));
  reg << count + count;
  reg << count + count;

  root.dump();
  assert(succeeded(root.verify()));

  return 0;
}