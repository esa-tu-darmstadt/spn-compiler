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

  PortInfo port(const std::string& name, PortDirection direction, Type type) {
    return PortInfo{
      .name = builder.getStringAttr(name),
      .direction = direction,
      .type = type
    };
  }

  StringAttr str(const std::string& s) { return StringAttr::get(ctxt, s); }
  MLIRContext *getContext() const { return ctxt; }

  Type getDefaultInputIndexType() {
    return builder.getI32Type();
  }

  Type getDefaultProbabilityType() {
    return builder.getF64Type();
  }

  Type getInputIndexType() {
    return IntType::get(builder.getI32IntegerAttr(8));
  }

  Type getProbabilityType() {
    // 64 bit float for now
    return IntType::get(builder.getI32IntegerAttr(64));
  }

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

    addArgumentMaterialization([hwFloatType](OpBuilder& builder, Type resultType, ValueRange inputs, Location loc) -> Optional<Value> {
      // just force a different type
      Value singleValue = inputs[0];
      singleValue.setType(resultType);
      return singleValue;
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

// type conversion pass

class HWTypeConverter : public TypeConverter {
public:
  explicit HWTypeConverter(PassHelper& helper) {
    addConversion([&helper](IntegerType indexType) {
      return helper.getInputIndexType();
    });

    addConversion([&helper](FloatType probType) {
      return helper.getProbabilityType();
    });
  }
};

class InstanceOpConversionPattern : public OpConversionPattern<InstanceOp> {
public:
  using OpConversionPattern<InstanceOp>::OpConversionPattern;

  //HWConversionPattern(HWTypeConverter& tc, MLIRContext *ctxt):
  //  ConversionPattern::ConversionPattern(tc, SPNLog::getOperationName(), 1, ctxt) {}

  LogicalResult matchAndRewrite(InstanceOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    //for (Value operand : adaptor.getOperands())
    //  operand.setType(typeConverter->convertType(operand.getType()));

    //for (OpResult result : op->getResults())
    //  result.setType(typeConverter->convertType(result.getType()));

    //ValueRange

    auto operands = adaptor.getOperands();
    std::vector<Value> _operands(std::begin(operands), std::end(operands));

    auto first = operands[0];

    rewriter.replaceOpWithNewOp<InstanceOp>(op,
      op.getReferencedModule(nullptr),
      op.getName(),
      //rewriter.getStringAttr("instance_" + std::to_string(id)),
      ArrayRef<Value>(_operands)
    );

    return success();
  }
};

class ModuleExternOpConversionPattern : public OpConversionPattern<HWModuleExternOp> {
  Type oldIndexType, oldProbType, newIndexType, newProbType;
public:
  using OpConversionPattern<HWModuleExternOp>::OpConversionPattern;

  ModuleExternOpConversionPattern(PassHelper& helper, TypeConverter& typeConverter, MLIRContext *ctxt):
    OpConversionPattern<HWModuleExternOp>::OpConversionPattern(typeConverter, ctxt) {
    oldIndexType = helper.getDefaultInputIndexType();
    oldProbType = helper.getDefaultProbabilityType();
    newIndexType = helper.getInputIndexType();
    newProbType = helper.getProbabilityType();
  }

  LogicalResult matchAndRewrite(HWModuleExternOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    std::vector<PortInfo> newPorts;

    for (PortInfo newPortInfo : op.getAllPorts()) {
      if (newPortInfo.type == oldIndexType)
        newPortInfo.type = newIndexType;
      else
        newPortInfo.type = newProbType;

      newPorts.push_back(newPortInfo);
    }

    HWModuleExternOp newMod = rewriter.replaceOpWithNewOp<HWModuleExternOp>(op,
      op.getNameAttr(),
      newPorts
    );

    

    return success();
  }
};

void convert(ModuleOp modOp);

}