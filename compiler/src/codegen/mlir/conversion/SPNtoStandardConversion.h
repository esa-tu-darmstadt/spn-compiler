//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_SPNTOSTANDARDCONVERSION_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_SPNTOSTANDARDCONVERSION_H

#include "../MLIRPassPipeline.h"
#include <driver/GlobalOptions.h>

namespace spnc {

  ///
  /// Action performing a (partial) conversion from SPN MLIR dialect to Standard dialect.
  struct SPNtoStandardConversion : public MLIRPipelineBase<SPNtoStandardConversion> {

  public:

    SPNtoStandardConversion(ActionWithOutput<mlir::ModuleOp>& input, std::shared_ptr<mlir::MLIRContext> ctx,
                            std::shared_ptr<mlir::ScopedDiagnosticHandler> handler, bool cpuVectorize);

    void initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx);

  private:

    bool vectorize;

  };
}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_CONVERSION_SPNTOSTANDARDCONVERSION_H
