//
// Created by ls on 2/7/20.
//

#ifndef SPNC_SPNDIALECT_H
#define SPNC_SPNDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"


namespace mlir {
    namespace spn {


        class SPNDialect : public mlir::Dialect {
        public:
            explicit SPNDialect(mlir::MLIRContext *ctx);

            /// Provide a utility accessor to the dialect namespace. This is used by
            /// several utilities for casting between dialects.
            static llvm::StringRef getDialectNamespace() { return "spn"; }
        };

        #define GET_OP_CLASSES
        #include "src/codegen/mlir/dialects/spn/SPNOps.h.inc"

    } // end namespace spn
} // end namespace mlir

#endif //SPNC_SPNDIALECT_H
