//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPN/LoSPNInterfaces.h"
#include "LoSPNtoCPU/LoSPNtoCPUConversionPasses.h"
#include "LoSPNtoCPU/LoSPNtoCPUTypeConverter.h"
#include "LoSPNtoCPU/StructurePatterns.h"
#include "LoSPNtoCPU/NodePatterns.h"
#include "LoSPNtoCPU/Vectorization/VectorizationPatterns.h"
#include "LoSPNtoCPU/Vectorization/LoSPNVectorizationTypeConverter.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"

void mlir::spn::LoSPNtoCPUStructureConversionPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalDialect<mlir::scf::SCFDialect>();
  target.addLegalDialect<mlir::vector::VectorDialect>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
  target.addLegalOp<FuncOp>();

  LoSPNtoCPUTypeConverter typeConverter;

  target.addLegalDialect<mlir::spn::low::LoSPNDialect>();
  target.addIllegalOp<mlir::spn::low::SPNKernel>();
  target.addIllegalOp<mlir::spn::low::SPNTask, mlir::spn::low::SPNBody>();

  OwningRewritePatternList patterns;
  if (vectorize) {
    // Try to vectorize tasks if vectorization was requested.
    mlir::spn::populateLoSPNCPUVectorizationStructurePatterns(patterns, &getContext(), typeConverter);
  }
  mlir::spn::populateLoSPNtoCPUStructurePatterns(patterns, &getContext(), typeConverter);

  FrozenRewritePatternList frozenPatterns(std::move(patterns));
  if (failed(applyPartialConversion(getOperation(), target, frozenPatterns))) {
    signalPassFailure();
  }

}

std::unique_ptr<mlir::Pass> mlir::spn::createLoSPNtoCPUStructureConversionPass(bool enableVectorization) {
  return std::make_unique<LoSPNtoCPUStructureConversionPass>(enableVectorization);
}

void mlir::spn::LoSPNtoCPUNodeConversionPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalDialect<mlir::scf::SCFDialect>();
  target.addLegalDialect<mlir::math::MathDialect>();
  // Linalg is required here, because we lower spn.copy to linalg.copy
  // as the Standard dialect currently does not have a copy operation.
  target.addLegalDialect<mlir::linalg::LinalgDialect>();
  target.addLegalDialect<mlir::vector::VectorDialect>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
  target.addLegalOp<FuncOp>();

  LoSPNtoCPUTypeConverter typeConverter;

  target.addIllegalDialect<mlir::spn::low::LoSPNDialect>();

  OwningRewritePatternList patterns;
  mlir::spn::populateLoSPNtoCPUNodePatterns(patterns, &getContext(), typeConverter);

  auto op = getOperation();
  FrozenRewritePatternList frozenPatterns(std::move(patterns));
  if (failed(applyFullConversion(op, target, frozenPatterns))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> mlir::spn::createLoSPNtoCPUNodeConversionPass() {
  return std::make_unique<LoSPNtoCPUNodeConversionPass>();
}

void mlir::spn::LoSPNNodeVectorizationPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalDialect<mlir::scf::SCFDialect>();
  target.addLegalDialect<mlir::math::MathDialect>();
  // Linalg is required here, because we lower spn.copy to linalg.copy
  // as the Standard dialect currently does not have a copy operation.
  target.addLegalDialect<mlir::linalg::LinalgDialect>();
  target.addLegalDialect<mlir::vector::VectorDialect>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
  target.addLegalOp<FuncOp>();

  // Walk the operation to find out which vector width to use for the type-converter.
  unsigned VF = 0;
  getOperation()->walk([&VF](low::LoSPNVectorizable vOp) {
    auto opVF = vOp.getVectorWidth();
    if (!VF && opVF) {
      VF = opVF;
    } else {
      if (opVF && VF != opVF) {
        vOp.getOperation()->emitOpError("Multiple vectorizations with differing vector width found");
      }
    }
  });
  // Use a special type converter converting all integer and floating-point types
  // to vectors of the type.
  LoSPNVectorizationTypeConverter typeConverter(VF);

  target.addDynamicallyLegalDialect<mlir::spn::low::LoSPNDialect>([](Operation* op) {
    if (auto vOp = dyn_cast<mlir::spn::low::LoSPNVectorizable>(op)) {
      if (vOp.checkVectorized()) {
        return false;
      }
    }
    return true;
  });
  // Mark ConvertToVector as legal. We will try to replace them during the conversion of
  // the remaining (scalar) nodes, as we need the scalar type to be legal, otherwise
  // the operand of ConvertToVector is converted to a vector before invoking the pattern.
  target.addLegalOp<mlir::spn::low::SPNConvertToVector>();

  OwningRewritePatternList patterns;
  mlir::spn::populateLoSPNCPUVectorizationNodePatterns(patterns, &getContext(), typeConverter);

  auto op = getOperation();
  FrozenRewritePatternList frozenPatterns(std::move(patterns));
  if (failed(applyPartialConversion(op, target, frozenPatterns))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> mlir::spn::createLoSPNNodeVectorizationPass() {
  return std::make_unique<LoSPNNodeVectorizationPass>();
}
