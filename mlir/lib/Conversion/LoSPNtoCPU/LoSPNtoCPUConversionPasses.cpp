//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPN/LoSPNInterfaces.h"
#include "LoSPNtoCPU/LoSPNtoCPUConversionPasses.h"
#include "LoSPNtoCPU/LoSPNtoCPUTypeConverter.h"
#include "LoSPNtoCPU/StructurePatterns.h"
#include "LoSPNtoCPU/NodePatterns.h"
#include "LoSPNtoCPU/Vectorization/VectorizationPatterns.h"
#include "LoSPNtoCPU/Vectorization/LoSPNVectorizationTypeConverter.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

void mlir::spn::LoSPNtoCPUStructureConversionPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalDialect<mlir::scf::SCFDialect>();
  target.addLegalDialect<mlir::vector::VectorDialect>();
  target.addLegalDialect<mlir::memref::MemRefDialect>();
  target.addLegalOp<ModuleOp>();
  target.addLegalOp<FuncOp>();

  LoSPNtoCPUTypeConverter typeConverter;

  target.addLegalDialect<mlir::spn::low::LoSPNDialect>();
  target.addIllegalOp<mlir::spn::low::SPNKernel>();
  target.addIllegalOp<mlir::spn::low::SPNTask, mlir::spn::low::SPNBody>();

  OwningRewritePatternList patterns(&getContext());
  if (vectorize) {
    // Try to vectorize tasks if vectorization was requested.
    mlir::spn::populateLoSPNCPUVectorizationStructurePatterns(patterns, &getContext(), typeConverter);
  }
  mlir::spn::populateLoSPNtoCPUStructurePatterns(patterns, &getContext(), typeConverter);

  auto op = getOperation();
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  if (failed(applyPartialConversion(op, target, frozenPatterns))) {
    signalPassFailure();
  }

}
void mlir::spn::LoSPNtoCPUStructureConversionPass::getDependentDialects(mlir::DialectRegistry& registry) const {
  registry.insert<StandardOpsDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::vector::VectorDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
}

std::unique_ptr<mlir::Pass> mlir::spn::createLoSPNtoCPUStructureConversionPass(bool enableVectorization) {
  return std::make_unique<LoSPNtoCPUStructureConversionPass>(enableVectorization);
}

void mlir::spn::LoSPNtoCPUNodeConversionPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalDialect<mlir::scf::SCFDialect>();
  target.addLegalDialect<mlir::math::MathDialect>();
  target.addLegalDialect<mlir::vector::VectorDialect>();
  target.addLegalDialect<mlir::memref::MemRefDialect>();
  target.addLegalOp<ModuleOp>();
  target.addLegalOp<FuncOp>();

  LoSPNtoCPUTypeConverter typeConverter;

  target.addIllegalDialect<mlir::spn::low::LoSPNDialect>();

  OwningRewritePatternList patterns(&getContext());
  mlir::spn::populateLoSPNtoCPUNodePatterns(patterns, &getContext(), typeConverter);

  auto op = getOperation();
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  if (failed(applyFullConversion(op, target, frozenPatterns))) {
    signalPassFailure();
  }
}
void mlir::spn::LoSPNtoCPUNodeConversionPass::getDependentDialects(mlir::DialectRegistry& registry) const {
  registry.insert<StandardOpsDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<mlir::vector::VectorDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
}

std::unique_ptr<mlir::Pass> mlir::spn::createLoSPNtoCPUNodeConversionPass() {
  return std::make_unique<LoSPNtoCPUNodeConversionPass>();
}

void mlir::spn::LoSPNNodeVectorizationPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalDialect<mlir::scf::SCFDialect>();
  target.addLegalDialect<mlir::math::MathDialect>();
  target.addLegalDialect<mlir::vector::VectorDialect>();
  target.addLegalDialect<mlir::memref::MemRefDialect>();
  target.addLegalOp<ModuleOp>();
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

  OwningRewritePatternList patterns(&getContext());
  mlir::spn::populateLoSPNCPUVectorizationNodePatterns(patterns, &getContext(), typeConverter);

  auto op = getOperation();
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  if (failed(applyPartialConversion(op, target, frozenPatterns))) {
    signalPassFailure();
  }
}
void mlir::spn::LoSPNNodeVectorizationPass::getDependentDialects(mlir::DialectRegistry& registry) const {
  registry.insert<StandardOpsDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<mlir::vector::VectorDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
}

std::unique_ptr<mlir::Pass> mlir::spn::createLoSPNNodeVectorizationPass() {
  return std::make_unique<LoSPNNodeVectorizationPass>();
}
