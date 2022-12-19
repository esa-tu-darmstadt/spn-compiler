#pragma once

#include <unordered_map>

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"

#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"
#include "HiSPN/HiSPNDialect.h"


using namespace ::mlir;
using namespace ::circt::hw;
using namespace ::mlir::spn::low;
using namespace ::mlir::spn::high;


namespace lo2hw {

class PassHelper {
public:
  HWModuleExternOp hwAddOp, hwMulOp, hwCatOp, hwConstOp, hwLogOp, hwBodyOp;
  MLIRContext *ctxt;
  OpBuilder builder;

  PassHelper(MLIRContext *ctxt): ctxt(ctxt), builder(ctxt) {}

  PortInfo port(const std::string& name, PortDirection direction, Type type);
  StringAttr str(const std::string& s) { return StringAttr::get(ctxt, s); }
  MLIRContext *getContext() const { return ctxt; }

  Type getIntType() { return builder.getI32Type(); }
  Type getFloatType() { return builder.getI64Type(); }

  template <class ConcreteOp>
  HWModuleExternOp getMod() const;
};

class SPNTypeConverter : public TypeConverter {
public:
  explicit SPNTypeConverter(Type hwIntType, Type hwFloatType) {
    addConversion([hwFloatType](FloatType floatType) -> Optional<Type> {
      return hwFloatType;
    });

    // TODO: What's the difference between IntType and IntegerType?
    addConversion([hwIntType](IntegerType intType) -> Optional<Type> {
      return hwIntType;
    });

    addSourceMaterialization([hwFloatType](OpBuilder& builder, FloatType resultType, ValueRange inputs, Location loc) -> Optional<Value> {
      builder.getIntegerAttr(
        builder.getI32Type(),
        123456
      );
      
      
      llvm::outs() << "?\n";
    });
  }
};

template <class ConcreteOp>
class SPNOpConversionPattern : public OpConversionPattern<ConcreteOp> {
  PassHelper helper;
public:
  using OpAdaptor = typename OpConversionPattern<ConcreteOp>::OpAdaptor;

  SPNOpConversionPattern(TypeConverter& typeConverter, MLIRContext *ctxt, PassHelper& helper):
    OpConversionPattern<ConcreteOp>::OpConversionPattern(typeConverter, ctxt),
    helper(helper) {}

  LogicalResult matchAndRewrite(ConcreteOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();

    HWModuleExternOp mod = helper.getMod<ConcreteOp>();

    uint64_t id = llvm::dyn_cast<IntegerAttr>(op.getOperation()->getAttr("instance_id")).getInt();

    std::vector<Value> _operands(std::begin(operands), std::end(operands));
    InstanceOp inst = rewriter.replaceOpWithNewOp<InstanceOp>(op.getOperation(),
        mod.getOperation(),
        rewriter.getStringAttr("instance_" + std::to_string(id)),
        ArrayRef<Value>(_operands)
    );

    return success();
  }
};

class SPNAddConversionPattern : public SPNOpConversionPattern<SPNAdd> {
public:
  using SPNOpConversionPattern<SPNAdd>::SPNOpConversionPattern;
};

class SPNMulConversionPattern : public SPNOpConversionPattern<SPNMul> {
public:
  using SPNOpConversionPattern<SPNMul>::SPNOpConversionPattern;
};

class SPNConstantConversionPattern : public SPNOpConversionPattern<SPNConstant> {
public:
  using SPNOpConversionPattern<SPNConstant>::SPNOpConversionPattern;
};

class SPNCategoricalLeafConversionPattern : public SPNOpConversionPattern<SPNCategoricalLeaf> {
public:
  using SPNOpConversionPattern<SPNCategoricalLeaf>::SPNOpConversionPattern;
};

class SPNLogConversionPattern : public SPNOpConversionPattern<SPNLog> {
public:
  using SPNOpConversionPattern<SPNLog>::SPNOpConversionPattern;
};

class SPNBodyConversionPattern : public SPNOpConversionPattern<SPNBody> {
public:
  using SPNOpConversionPattern<SPNBody>::SPNOpConversionPattern;
};

//class SPNBodyConversionPattern : public OpConversionPattern<SPNBody> {
//public:
//  using OpConversionPattern<SPNBody>::OpConversionPattern;
//
//  LogicalResult matchAndRewrite(SPNBody op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
//
//  }
//};

class SPNYieldConversionPattern : public OpConversionPattern<SPNYield> {
public:
  using OpConversionPattern<SPNYield>::OpConversionPattern;

  LogicalResult matchAndRewrite(SPNYield op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    // adaptor is the already converted result
    auto operand = adaptor.getOperands()[0];

    rewriter.replaceOpWithNewOp<OutputOp>(op,
      ValueRange{operand}
    );

    return success();
  }
};

static PortInfo port(const std::string& name, PortDirection direction, Type type);
static void prepare(ModuleOp modOp, PassHelper& helper);

void convert(ModuleOp modOp);

static void test(MLIRContext *ctxt);

}