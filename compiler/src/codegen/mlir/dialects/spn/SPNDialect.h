//
// Created by ls on 2/7/20.
//

#ifndef SPNC_SPNDIALECT_H
#define SPNC_SPNDIALECT_H

#include <mlir/IR/Dialect.h>
#include <mlir/IR/Function.h>
#include "SPNOpInterfaces.h"

namespace mlir {

  ///
  /// Sub-namespace of namespace "mlir" for all components related to the MLIR toolchain.
  namespace spn {

    ///
    /// Dialect for SPN specific operations.
    class SPNDialect : public mlir::Dialect {
    public:
      /// Constructor.
      /// \param ctx Surrounding MLIR context.
      explicit SPNDialect(mlir::MLIRContext* ctx);

      /// Provide a utility accessor to the dialect namespace. This is used by
      /// several utilities for casting between dialects.
      static llvm::StringRef getDialectNamespace() { return "spn"; }
    };

// Include all the operation declarations.
#define GET_OP_CLASSES
#include "src/codegen/mlir/dialects/spn/SPNOps.h.inc"



} // end namespace spn

// Include dialect specific attributes.
#include "src/codegen/mlir/dialects/spn/SPNOps.attr.h.inc"
} // end namespace mlir

#endif //SPNC_SPNDIALECT_H
