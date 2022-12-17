#include "lo2hwPass.h"

/*
  What is a SPN task?
  

  SPN body represents a function of type i32, ..., i32 -> f64.
  This can be trivially represented as a hw module as hw does not make
  any assumptions about input/output types.

  */

namespace lo2hw {

PortInfo PassHelper::port(const std::string& name, PortDirection direction, Type type)
{
  return PortInfo{
    .name = StringAttr::get(ctxt, name),
    .direction = direction,
    .type = type
  };
}

template <> HWModuleExternOp PassHelper::getMod<SPNAdd>() const { return hwAddOp; }
template <> HWModuleExternOp PassHelper::getMod<SPNMul>() const { return hwMulOp; }
template <> HWModuleExternOp PassHelper::getMod<SPNConstant>() const { return hwConstOp; }
template <> HWModuleExternOp PassHelper::getMod<SPNCategoricalLeaf>() const { return hwCatOp; }
template <> HWModuleExternOp PassHelper::getMod<SPNLog>() const { return hwLogOp; }

LogicalResult SPNBodyConversionPattern::matchAndRewrite(SPNBody body, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const
{
  std::vector<PortInfo> ports{
    helper.port("out_prob", PortDirection::OUTPUT, rewriter.getF64Type())
  };

  int32_t i = 0;
  for (auto operand : adaptor.getOperands()) {
    ports.push_back(helper.port("in_" + std::to_string(i++), PortDirection::INPUT, rewriter.getI32Type()));
  }

  rewriter.replaceOpWithNewOp<OutputOp>(
    body.getOperation()->getRegion(0).front().getTerminator(),
    body.getOperation()->getRegion(0).front().getTerminator()->getResults()
  );

  HWModuleOp newOp = rewriter.replaceOpWithNewOp<HWModuleOp>(
    body,
    rewriter.getStringAttr("lalallala"),
    ArrayRef<PortInfo>(ports)
  );

  llvm::outs() << newOp.getNumResults() << "\n";

  return success();
}

void prepare(ModuleOp modOp, PassHelper& helper) {
  MLIRContext *ctxt = modOp.getContext();
  OpBuilder builder(ctxt);

  // TODO

  // assign unique instance ids (even if some aren't needed)
  int64_t uid = 0;

  modOp.walk([&](Operation *op) {
    op->setAttr("instance_id", builder.getI64IntegerAttr(uid++));
  });

  // create the hardware extern modules on top of the body
  std::vector<PortInfo> binPorts{
    helper.port("in_a", PortDirection::INPUT, builder.getF64Type()),
    helper.port("in_a", PortDirection::INPUT, builder.getF64Type()),
    helper.port("out_c", PortDirection::OUTPUT, builder.getF64Type())
  };

  std::vector<PortInfo> catPorts{
    helper.port("in_index", PortDirection::INPUT, builder.getIntegerType(8)),
    helper.port("out_prob", PortDirection::OUTPUT, builder.getF64Type())
  };

  std::vector<PortInfo> constPorts{
    helper.port("out_const", PortDirection::OUTPUT, builder.getF64Type())
  };

  std::vector<PortInfo> logPorts{
    helper.port("in_a", PortDirection::INPUT, builder.getF64Type()),
    helper.port("out_b", PortDirection::OUTPUT, builder.getF64Type())
  };

  helper.hwAddOp = builder.create<HWModuleExternOp>(
    builder.getUnknownLoc(),
    builder.getStringAttr("sv_fadd"),
    llvm::ArrayRef<PortInfo>(binPorts)
  );

  helper.hwMulOp = builder.create<HWModuleExternOp>(
    builder.getUnknownLoc(),
    builder.getStringAttr("sv_fmul"),
    llvm::ArrayRef<PortInfo>(binPorts)
  );

  helper.hwCatOp = builder.create<HWModuleExternOp>(
    builder.getUnknownLoc(),
    builder.getStringAttr("sv_fcategorical"),
    llvm::ArrayRef<PortInfo>(catPorts)
  );

  helper.hwConstOp = builder.create<HWModuleExternOp>(
    builder.getUnknownLoc(),
    builder.getStringAttr("sv_fconstant"),
    llvm::ArrayRef<PortInfo>(constPorts)
  );

  helper.hwLogOp = builder.create<HWModuleExternOp>(
    builder.getUnknownLoc(),
    builder.getStringAttr("sv_flog"),
    llvm::ArrayRef<PortInfo>(logPorts)
  );

  builder.setInsertionPointToStart(&modOp.getOperation()->getRegion(0).front());

  builder.insert(helper.hwAddOp.getOperation());
  builder.insert(helper.hwMulOp.getOperation());
  builder.insert(helper.hwCatOp.getOperation());
  builder.insert(helper.hwConstOp.getOperation());
  builder.insert(helper.hwLogOp.getOperation());
}

LogicalResult createModuleFromBody(SPNBody body, PassHelper& helper) {
  OpBuilder builder(helper.getContext());

  // go through all the computation nodes and replace them with instances
  //body.walk([&](LoSPN_Op op) {



  //});

  // trivially replace all the computation operations in the body with hw instances
  ConversionTarget target(*helper.getContext());

  target.addLegalDialect<LoSPNDialect>();
  target.addLegalDialect<HWDialect>();

  target.addIllegalOp<SPNAdd>();
  target.addIllegalOp<SPNMul>();
  target.addIllegalOp<SPNConstant>();
  target.addIllegalOp<SPNCategoricalLeaf>();
  target.addIllegalOp<SPNYield>();
  target.addIllegalOp<SPNLog>();

  RewritePatternSet patterns(helper.getContext());
  patterns.add<SPNAddConversionPattern>(helper.getContext(), helper);
  patterns.add<SPNMulConversionPattern>(helper.getContext(), helper);
  patterns.add<SPNConstantConversionPattern>(helper.getContext(), helper);
  patterns.add<SPNCategoricalLeafConversionPattern>(helper.getContext(), helper);
  patterns.add<SPNYieldConversionPattern>(helper.getContext());
  patterns.add<SPNLogConversionPattern>(helper.getContext(), helper);
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));

  if (failed(applyPartialConversion(body, target, frozenPatterns)))
    return failure();

  // create a new hw module and insert the new body
  std::vector<PortInfo> ports{
    helper.port("in_index", PortDirection::INPUT, builder.getI8Type())
  };

  for (uint32_t i = 0; i < body.getOperands().size(); ++i)
    ports.push_back(
      helper.port("out_" + std::to_string(i), PortDirection::OUTPUT, builder.getF64Type())
    );

  uint64_t instanceId = llvm::dyn_cast<IntegerAttr>(body.getOperation()->getAttr("instance_id")).getInt();
  std::string modName = "instance_" + std::to_string(instanceId);

  builder.setInsertionPointAfter(body.getOperation());
  HWModuleOp modOp = builder.create<HWModuleOp>(
    body.getLoc(),
    builder.getStringAttr(modName),
    ArrayRef<PortInfo>(ports)
  );

  // see FunctionInterfaces.td
  //modOp.eraseBody();
  //modOp.getBody().push_back(new Block);
  
  //modOp.getBodyBlock()->getOperations().splice(
  //  modOp.getBodyBlock()->getOperations().begin(), // where
  //  body.getOperation()->getBlock()->getOperations(),
  //  body.getOperation()->getBlock()->begin(),
  //  body.getOperation()->getBlock()->end()
  //);

  //body.getOperation()->getBlock()->getOperations().splice(
  //  modOp.getBody()
  //);

  //modOp.dump();

  return success();
}

void test(MLIRContext *ctxt) {
  OpBuilder builder(ctxt);
  PassHelper helper(ctxt);

  std::vector<PortInfo> ports{
    helper.port("in_index", PortDirection::INPUT, builder.getIntegerType(8)),
    helper.port("out_prob", PortDirection::OUTPUT, builder.getF64Type())
  };

  HWModuleOp newOp = builder.create<HWModuleOp>(
    builder.getUnknownLoc(),
    builder.getStringAttr("extract_cstfold"),
    ArrayRef<PortInfo>(ports)
  );

  builder.setInsertionPointToStart(newOp.getBodyBlock());
  ConstantOp constOp = builder.create<ConstantOp>(
    builder.getUnknownLoc(),
    builder.getIntegerType(16),
    420
  );

  newOp.getBodyBlock()->getTerminator()->erase();
  builder.setInsertionPointToEnd(newOp.getBodyBlock());
  builder.create<OutputOp>(
    builder.getUnknownLoc(),
    ValueRange{constOp.getResult()}
  );

  newOp.dump(); 
}

void convert(ModuleOp modOp) {
  //test(modOp.getContext());
  //return;

  PassHelper helper(modOp.getContext());
  
  prepare(modOp, helper);

  modOp.walk([&](SPNBody body) {
    std::ignore = createModuleFromBody(body, helper);
  });
}

}