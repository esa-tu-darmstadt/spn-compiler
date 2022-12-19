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
template <> HWModuleExternOp PassHelper::getMod<SPNBody>() const { return hwBodyOp; }

void prepare(ModuleOp modOp, PassHelper& helper) {
  MLIRContext *ctxt = modOp.getContext();
  OpBuilder builder(ctxt);

  // TODO

  // assign unique instance ids (even if some aren't needed)
  int64_t uid = 0;

  modOp.walk([&](Operation *op) {
    op->setAttr("instance_id", builder.getI64IntegerAttr(uid++));
  });

  Type intType = helper.getIntType();
  Type floatType = helper.getFloatType();

  // create the hardware extern modules on top of the body
  std::vector<PortInfo> binPorts{
    helper.port("in_a", PortDirection::INPUT, floatType),
    helper.port("in_a", PortDirection::INPUT, floatType),
    helper.port("out_c", PortDirection::OUTPUT, floatType)
  };

  std::vector<PortInfo> catPorts{
    helper.port("in_index", PortDirection::INPUT, intType),
    helper.port("out_prob", PortDirection::OUTPUT, floatType)
  };

  std::vector<PortInfo> constPorts{
    helper.port("out_const", PortDirection::OUTPUT, floatType)
  };

  std::vector<PortInfo> logPorts{
    helper.port("in_a", PortDirection::INPUT, floatType),
    helper.port("out_b", PortDirection::OUTPUT, floatType)
  };

  std::vector<PortInfo> bodyPorts{
    helper.port("out_prob", PortDirection::OUTPUT, floatType)
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

  helper.hwBodyOp = builder.create<HWModuleExternOp>(
    builder.getUnknownLoc(),
    builder.getStringAttr("sv_body"),
    llvm::ArrayRef<PortInfo>(bodyPorts)
  );
}

static Optional<HWModuleOp> createModuleFromBody(ModuleOp root, SPNBody body, PassHelper& helper) {
  OpBuilder builder(helper.getContext());

  // remove attribute, otherwise verifier is mad
  root.getOperation()->removeAttr("instance_id");

  // create hw module from scratch
  builder.clearInsertionPoint();

  std::vector<PortInfo> modOpPorts{
    helper.port("out_prob", PortDirection::OUTPUT, helper.getFloatType())
  };

  for (std::size_t i = 0; i < body.getOperands().size(); ++i)
    modOpPorts.push_back(helper.port(
      "in_" + std::to_string(i), PortDirection::INPUT, helper.getIntType()
    ));

  HWModuleOp modOp = builder.create<HWModuleOp>(
    builder.getUnknownLoc(),
    builder.getStringAttr("spn_body_mod"),
    ArrayRef<PortInfo>(modOpPorts)
  );

  // rewrite
  {
    ConversionTarget target(*helper.getContext());

    target.addLegalDialect<LoSPNDialect>();
    target.addLegalDialect<HWDialect>();

    target.addIllegalOp<SPNAdd>();
    target.addIllegalOp<SPNMul>();
    target.addIllegalOp<SPNConstant>();
    target.addIllegalOp<SPNCategoricalLeaf>();
    target.addIllegalOp<SPNYield>();
    target.addIllegalOp<SPNLog>();
    //target.addIllegalOp<SPNBody>();

    SPNTypeConverter typeConverter(
      helper.getIntType(),
      helper.getFloatType()
    );

    RewritePatternSet patterns(helper.getContext());
    patterns.add<SPNAddConversionPattern>(typeConverter, helper.getContext(), helper);
    patterns.add<SPNMulConversionPattern>(typeConverter, helper.getContext(), helper);
    patterns.add<SPNConstantConversionPattern>(typeConverter, helper.getContext(), helper);
    patterns.add<SPNCategoricalLeafConversionPattern>(typeConverter, helper.getContext(), helper);
    patterns.add<SPNYieldConversionPattern>(helper.getContext());
    patterns.add<SPNLogConversionPattern>(typeConverter, helper.getContext(), helper);
    //patterns.add<SPNBodyConversionPattern>(helper.getContext(), helper);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    if (failed(applyPartialConversion(body, target, frozenPatterns)))
      return Optional<HWModuleOp>();
  }
 
  // delete hw output, see FunctionInterfaces.td
  modOp.eraseBody();
  modOp.getBody().push_back(new Block);

  builder.setInsertionPointToStart(modOp.getBodyBlock());

  // region, first block
  auto& bodyOps = body.getBody().front().getOperations();

  //modOp.getBodyBlock()->getOperations().splice(
  //  modOp.getBodyBlock()->getOperations().begin(), // where to insert
  //  bodyOps, bodyOps.begin(), bodyOps.end() // where to take from
  //);

  // performs all the cleanup that I'm too stupid to do
  modOp.getRegion().takeBody(body.getRegion());

  // insert external hw modules and hw module
  builder.setInsertionPointAfter(&root.front());
  builder.insert(helper.hwAddOp);
  builder.insert(helper.hwMulOp);
  builder.insert(helper.hwCatOp);
  builder.insert(helper.hwConstOp);
  builder.insert(helper.hwLogOp);
  builder.insert(helper.hwBodyOp);
  builder.insert(modOp);

  // delete the first operation i.e. remaining spn stuff
  root.getRegion().front().front().erase();

  return modOp;
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

  newOp.appendInput("hello", builder.getI8Type());
  llvm::outs() << "result count: " << newOp.getResultTypes().size() << "\n";

  newOp.dump(); 
}

void convert(ModuleOp modOp) {
  //test(modOp.getContext());
  //return;

  PassHelper helper(modOp.getContext());
  
  prepare(modOp, helper);

  modOp.walk([&](SPNBody body) {
    createModuleFromBody(modOp, body, helper);
    return WalkResult::interrupt();
  });
}

}