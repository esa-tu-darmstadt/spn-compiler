#include "lo2hwPass.h"

#include <mlir/IR/BuiltinDialect.h>
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

  Type indexType = helper.getInputIndexType();
  Type floatType = helper.getProbabilityType();

  // create the hardware extern modules on top of the body
  std::vector<PortInfo> binPorts{
    helper.port("in_a", PortDirection::INPUT, floatType),
    helper.port("in_a", PortDirection::INPUT, floatType),
    helper.port("out_c", PortDirection::OUTPUT, floatType)
  };

  std::vector<PortInfo> catPorts{
    helper.port("in_index", PortDirection::INPUT, indexType),
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
    helper.port("out_prob", PortDirection::OUTPUT, helper.getProbabilityType())
  };

  for (std::size_t i = 0; i < body.getOperands().size(); ++i)
    modOpPorts.push_back(helper.port(
      "in_" + std::to_string(i), PortDirection::INPUT, helper.getInputIndexType()
    ));

  HWModuleOp modOp = builder.create<HWModuleOp>(
    builder.getUnknownLoc(),
    builder.getStringAttr("spn_body_mod"),
    ArrayRef<PortInfo>(modOpPorts)
  );

  // delete hw output, see FunctionInterfaces.td
  modOp.eraseBody();
  modOp.getBody().push_back(new Block);

  builder.setInsertionPointToStart(modOp.getBodyBlock());

  // region, first block
  auto& bodyOps = body.getBody().front().getOperations();

  // performs all the cleanup that I'm too stupid to do
  modOp.getRegion().takeBody(body.getRegion());

  builder.setInsertionPointAfter(&root.front());
  builder.insert(modOp);

  // delete the first operation i.e. remaining spn stuff
  root.getRegion().front().front().erase();

  // insert the external modules with the final types
  builder.setInsertionPoint(&root.front());
  builder.insert(helper.hwAddOp);
  builder.insert(helper.hwMulOp);
  builder.insert(helper.hwCatOp);
  builder.insert(helper.hwConstOp);
  builder.insert(helper.hwLogOp);
  builder.insert(helper.hwBodyOp);

  // rewrite and type convert
  {
    ConversionTarget target(*helper.getContext());

    target.addLegalDialect<LoSPNDialect>();
    target.addLegalDialect<HWDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addLegalDialect<BuiltinDialect>();

    target.addIllegalOp<SPNAdd>();
    target.addIllegalOp<SPNMul>();
    target.addIllegalOp<SPNConstant>();
    target.addIllegalOp<SPNCategoricalLeaf>();
    target.addIllegalOp<SPNYield>();
    target.addIllegalOp<SPNLog>();
    //target.addIllegalOp<SPNBody>();

    SPNTypeConverter typeConverter(
      helper.getInputIndexType(),
      helper.getProbabilityType()
    );

    RewritePatternSet patterns(helper.getContext());
    patterns.add<SPNAddConversionPattern>(typeConverter, helper.getContext(), helper);
    patterns.add<SPNMulConversionPattern>(typeConverter, helper.getContext(), helper);
    patterns.add<SPNConstantConversionPattern>(typeConverter, helper.getContext(), helper);
    patterns.add<SPNCategoricalLeafConversionPattern>(typeConverter, helper.getContext(), helper);
    patterns.add<SPNYieldConversionPattern>(typeConverter, helper.getContext());
    patterns.add<SPNLogConversionPattern>(typeConverter, helper.getContext(), helper);
    //patterns.add<SPNBodyConversionPattern>(helper.getContext(), helper);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    if (failed(applyPartialConversion(modOp, target, frozenPatterns)))
      return Optional<HWModuleOp>();
  }

  return modOp;
}

void convert(ModuleOp modOp) {
  PassHelper helper(modOp.getContext());
  
  prepare(modOp, helper);

  modOp.walk([&](SPNBody body) {
    createModuleFromBody(modOp, body, helper);
    return WalkResult::interrupt();
  });
}

}