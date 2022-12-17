/**
 * This is performs manual handwritten conversion of the lospn tree to the hw tree.
 * This is merely and exercise to get familiar with the inner workings of MLIR and CIRCT.
 * Inspiration was mainly drawn from MLIRGen from the toy dialect.
*/

#include "mlirgen.hpp"


mlir::OwningOpRef<mlir::ModuleOp> mlir_gen(mlir::MLIRContext *context, mlir::ModuleOp *root)
{
    return mlir_gen_impl(context).gen(root);
}