#include "LoSPNtoFPGA/DelayInsertionPass.hpp"


void mlir::spn::fpga::DelayInsertionPass::getDependentDialects(DialectRegistry& registry) const {
  registry.insert<circt::hw::HWDialect>();
  registry.insert<circt::seq::SeqDialect>();
}

void mlir::spn::fpga::DelayInsertionPass::runOnOperation() {

  Operation *op = getOperation();
  MLIRContext *ctxt = op->getContext();
  OpBuilder builder(ctxt);

  Value clk, rst;

  // stupid way to find the clk and rst signals
  op->walk([&](HWModuleOp op) {
    Block *body = op.getBodyBlock();
    clk = body->getArguments()[0];
    rst = body->getArguments()[1];
    return WalkResult::interrupt();
  });

  auto delaySignal = [&](Value input, uint32_t delay) -> std::tuple<Value, Operation *> {
    assert(delay >= 1);

    builder.setInsertionPointAfter(input.getDefiningOp());
    Value prev = input;
    Operation *ignoreOp = nullptr;

    for (uint32_t i = 0; i < delay; ++i) {
      //CompRegOp reg = builder.create<CompRegOp>(
      //  builder.getUnknownLoc(), std::move(prev), clk, "shiftReg"
      //);
      FirRegOp reg = builder.create<FirRegOp>(
        builder.getUnknownLoc(), std::move(prev), clk, builder.getStringAttr("shiftReg")
      );
      prev = reg.getResult();

      if (!ignoreOp)
        ignoreOp = reg;
    }

    return std::make_tuple(prev, ignoreOp);
  };

  for (auto [from, to, delay] : schedule.delays) {
    Value result = from->getResults()[0];
    auto [delayedResult, ignoreOp] = delaySignal(result, delay);
    // we introduce a new usage which we need to ignore
    result.replaceAllUsesExcept(delayedResult, ignoreOp);
  }

}