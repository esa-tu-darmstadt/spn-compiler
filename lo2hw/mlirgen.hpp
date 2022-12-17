#ifndef MLIRGEN_HPP
#define MLIRGEN_HPP

#include <iostream>
#include <memory>
#include <string>
#include <cassert>

#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"

#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"
#include "HiSPN/HiSPNDialect.h"


class mlir_gen_impl
{
    mlir::OpBuilder builder;
public:
    mlir_gen_impl(mlir::MLIRContext *context): builder(context) {}

    mlir::ModuleOp gen(mlir::ModuleOp *op)
    {
        walk_operation(op->getOperation());

        mlir::ModuleOp mod = mlir::ModuleOp::create(builder.getUnknownLoc());
        return mod;
    }

    
private:

    void gen(mlir::spn::low::SPNAdd op)
    {
        mlir::Location loc = op.getLoc();

        std::vector<circt::hw::PortInfo> ports{

        };

        // TODO: String to StringAttr ???
        mlir::StringAttr name = mlir::StringAttr::get(builder.getContext(), "sv_my_adder");

        circt::hw::HWModuleExternOp hw_op = builder.create<circt::hw::HWModuleExternOp>(
            loc,
            name,
            llvm::ArrayRef<circt::hw::PortInfo>(ports)
        );

        hw_op.dump();
    }

    void walk_operation(mlir::Operation *op)
    {
        if (llvm::isa<mlir::spn::low::SPNAdd>(op)) {
            llvm::outs() << "ADDER!\n";
            gen(llvm::dyn_cast<mlir::spn::low::SPNAdd>(op));
        }

        llvm::outs() << "visiting op: '" << op->getName() << "' with "
                    << op->getNumOperands() << " operands and "
                    << op->getNumResults() << " results\n";

        for (mlir::Region& region : op->getRegions())
            walk_region(region);

    }

    void walk_region(mlir::Region& region)
    {
        llvm::outs() << "Region with " << region.getBlocks().size()
                    << " blocks:\n";

        for (mlir::Block& block : region.getBlocks()) {
            walk_block(block);
        }
    }

    void walk_block(mlir::Block& block)
    {
        llvm::outs()
            << "Block with " << block.getNumArguments() << " arguments, "
            << block.getNumSuccessors()
            << " successors, and "
            // Note, this `.size()` is traversing a linked-list and is O(n).
            << block.getOperations().size() << " operations\n";

        for (mlir::Operation& op : block.getOperations())
            walk_operation(&op);
    }

};

mlir::OwningOpRef<mlir::ModuleOp> mlir_gen(mlir::MLIRContext *context, mlir::ModuleOp *root);

#endif