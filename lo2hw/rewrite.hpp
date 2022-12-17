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



class ExternHWModules
{
    MLIRContext *context;
    HWModuleExternOp hwAdd, hwMul, hwCategorical, hwConstant;

    PortInfo port(const std::string& name, PortDirection direction, Type type) const
    {
        return PortInfo{
            .name = StringAttr::get(context, name),
            .direction = direction,
            .type = type
        };
    }
public:
    ExternHWModules(MLIRContext *context, Operation *op):
        context(context)
    {
        OpBuilder builder(context);

        std::vector<PortInfo> bin_ports{
            port("in_a", PortDirection::INPUT, builder.getF64Type()),
            port("in_a", PortDirection::INPUT, builder.getF64Type()),
            port("out_c", PortDirection::OUTPUT, builder.getF64Type())
        };

        std::vector<PortInfo> cat_ports{
            port("in_index", PortDirection::INPUT, builder.getIntegerType(8)),
            port("out_prob", PortDirection::OUTPUT, builder.getF64Type())
        };

        std::vector<PortInfo> const_ports{
            port("out_const", PortDirection::OUTPUT, builder.getF64Type())
        };

        hwAdd = builder.create<HWModuleExternOp>(
            builder.getUnknownLoc(),
            builder.getStringAttr("sv_fadd"),
            llvm::ArrayRef<PortInfo>(bin_ports)
        );

        hwMul = builder.create<HWModuleExternOp>(
            builder.getUnknownLoc(),
            builder.getStringAttr("sv_fmul"),
            llvm::ArrayRef<PortInfo>(bin_ports)
        );

        hwCategorical = builder.create<HWModuleExternOp>(
            builder.getUnknownLoc(),
            builder.getStringAttr("sv_categorical"),
            llvm::ArrayRef<PortInfo>(cat_ports)
        );

        hwConstant = builder.create<HWModuleExternOp>(
            builder.getUnknownLoc(),
            builder.getStringAttr("sv_constant"),
            llvm::ArrayRef<PortInfo>(const_ports)
        );

        builder.setInsertionPointToStart(
            &op->getRegion(0).front()
        );

        builder.insert(hwAdd.getOperation());
        builder.insert(hwMul.getOperation());
        builder.insert(hwCategorical.getOperation());
        builder.insert(hwConstant.getOperation());
    }

    HWModuleExternOp getAdd() const { return hwAdd; }
    HWModuleExternOp getMul() const { return hwMul; }
    HWModuleExternOp getCategorical() const { return hwCategorical; }
    HWModuleExternOp getConstant() const { return hwConstant; }
};



template <class SourceOp>
class lo2hwOp : public OpConversionPattern<SourceOp>
{
protected:
    MLIRContext *context;
    std::shared_ptr<ExternHWModules> externHWModules;

    std::unordered_map<std::string, HWModuleExternOp> mods;
public:
    lo2hwOp(MLIRContext *context, const std::shared_ptr<ExternHWModules>& externHWModules):
        OpConversionPattern<SourceOp>::OpConversionPattern(context),
        context(context),
        externHWModules(externHWModules) {}

    StringAttr str(const std::string& s) const
    {
        return StringAttr::get(context, s);
    }
public:
    using OpConversionPattern<SourceOp>::OpConversionPattern;
};

class lo2hwAdd : public lo2hwOp<SPNAdd>
{
public:
    using OpAdaptor = OpConversionPattern<SPNAdd>::OpAdaptor;

    lo2hwAdd(MLIRContext *ctxt, const std::shared_ptr<ExternHWModules>& externHWModules):
        lo2hwOp(ctxt, externHWModules) {}

    LogicalResult matchAndRewrite(SPNAdd op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        auto operands = adaptor.getOperands();
        assert(operands.size() == 2);

        HWModuleExternOp mod = externHWModules->getAdd();

        std::vector<Value> _operands(std::begin(operands), std::end(operands));
        InstanceOp fadd_inst = rewriter.replaceOpWithNewOp<InstanceOp>(op.getOperation(),
            mod.getOperation(), str("sv_fadd_inst"), ArrayRef<Value>(_operands)
        );

        return success();
    }
};

class lo2hwMul : public lo2hwOp<SPNMul>
{
public:
    using OpAdaptor = OpConversionPattern<SPNMul>::OpAdaptor;

    lo2hwMul(MLIRContext *ctxt, const std::shared_ptr<ExternHWModules>& externHWModules):
        lo2hwOp(ctxt, externHWModules) {}

    LogicalResult matchAndRewrite(SPNMul op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
    {
        auto operands = adaptor.getOperands();
        assert(operands.size() == 2);

        HWModuleExternOp mod = externHWModules->getMul();

        std::vector<Value> _operands(std::begin(operands), std::end(operands));
        InstanceOp fmul_inst = rewriter.replaceOpWithNewOp<InstanceOp>(op.getOperation(),
            mod.getOperation(), str("sv_fmul_inst"), ArrayRef<Value>(_operands)
        );

        return success();
    }
};

class lo2hwCategorical : public lo2hwOp<SPNCategoricalLeaf>
{
public:
    using OpAdaptor = OpConversionPattern<SPNCategoricalLeaf>::OpAdaptor;

    lo2hwCategorical(MLIRContext *ctxt, const std::shared_ptr<ExternHWModules>& externHWModules):
        lo2hwOp(ctxt, externHWModules) {}

    LogicalResult matchAndRewrite(SPNCategoricalLeaf op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
    {
        auto operands = adaptor.getOperands();
        assert(operands.size() == 1);

        HWModuleExternOp mod = externHWModules->getCategorical();

        std::vector<Value> _operands(std::begin(operands), std::end(operands));
        InstanceOp cat_inst = rewriter.replaceOpWithNewOp<InstanceOp>(op.getOperation(),
            mod.getOperation(), str("sv_categorical_inst"), ArrayRef<Value>(_operands)
        );

        return success();
    }
};

class lo2hwConstant : public lo2hwOp<SPNConstant>
{
public:
    using OpAdaptor = OpConversionPattern<SPNConstant>::OpAdaptor;
    using lo2hwOp<SPNConstant>::lo2hwOp;

    LogicalResult matchAndRewrite(SPNConstant op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
    {
        auto operands = adaptor.getOperands();
        assert(operands.size() == 0);

        HWModuleExternOp mod = externHWModules->getConstant();

        std::vector<Value> _operands(std::begin(operands), std::end(operands));
        InstanceOp const_inst = rewriter.replaceOpWithNewOp<InstanceOp>(op.getOperation(),
            mod.getOperation(), str("sv_constant_inst"), ArrayRef<Value>(_operands)
        );

        return success();
    }
};

inline LogicalResult applyLo2hw(MLIRContext *context, Operation *op)
{
    std::shared_ptr<ExternHWModules> externHWModules = std::make_shared<ExternHWModules>(context, op);

    ConversionTarget target(*context);

    target.addLegalDialect<LoSPNDialect>();
    target.addLegalDialect<HWDialect>();

    // mark operations that should be replaced illegal
    target.addIllegalOp<SPNAdd>();
    target.addIllegalOp<SPNMul>();
    target.addIllegalOp<SPNCategoricalLeaf>();
    target.addIllegalOp<SPNConstant>();

    RewritePatternSet patterns(context);
    patterns.add<lo2hwAdd>(context, externHWModules);
    patterns.add<lo2hwMul>(context, externHWModules);
    patterns.add<lo2hwCategorical>(context, externHWModules);
    patterns.add<lo2hwConstant>(context, externHWModules);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    return applyPartialConversion(op, target, frozenPatterns);
}
