#include <iostream>
#include <memory>
#include <string>
#include <cassert>

#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "LoSPN/LoSPNDialect.h"
#include "HiSPN/HiSPNDialect.h"

#include "mlirgen.hpp"
#include "rewrite.hpp"
#include "lo2hwPass.h"
#include "conversion.hpp"


void walk_operation(mlir::Operation *);
void walk_region(mlir::Region&);
void walk_block(mlir::Block&);

// walk AST and print
void walk_operation(mlir::Operation *op)
{
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

// walk AST and create new nodes

void dump_mlir(const std::string& src_file_path)
{
    //llvm::DebugFlag = true;

    std::unique_ptr<mlir::MLIRContext> context = std::make_unique<mlir::MLIRContext>();
    assert(context->getOrLoadDialect<mlir::spn::low::LoSPNDialect>());
    assert(context->getOrLoadDialect<mlir::spn::high::HiSPNDialect>());
    assert(context->getOrLoadDialect<circt::hw::HWDialect>());
    assert(context->getOrLoadDialect<circt::seq::SeqDialect>());

    mlir::ParserConfig parser_config(context.get());

    // somehow load the mlir file and get an AST
    //std::cout << "INFO: Trying to parse source file " << src_file_path << std::endl;
    mlir::OwningOpRef<mlir::Operation *> result = mlir::parseSourceFile<mlir::Operation *>(src_file_path, parser_config);
    mlir::Operation *op = result.get();
    assert(op);

    //llvm::outs() << "INFO: Got:\n";
    //op->dump();

    // walk the AST and print the nodes
    //llvm::outs() << "INFO: Walking:\n";
    //walk_operation(reinterpret_cast<mlir::Operation *>(op));

    // walk the AST and create new nodes in HW
    //llvm::outs() << "INFO: Converting...\n";
    //assert(applyLo2hw(context.get(), op).succeeded());
    //op->dump();
    //auto modOp = llvm::dyn_cast<ModuleOp>(op);
    //ModuleOp newRoot = ::spn::lo2hw::conversion::convert(modOp);
    //newRoot.dump();

    ::spn::lo2hw::conversion::test(context.get());
}



void print_usage()
{
    std::cout << "Usage: lo2hw <mlir file path>" << std::endl;
}

int main(int argc, const char **argv)
{
    if (argc <= 1) {
        print_usage();
        return 1;
    }

    dump_mlir(argv[1]);

    return 0;
}