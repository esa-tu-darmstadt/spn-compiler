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
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

void mlir::spn::LoSPNtoCPUStructureConversionPass::getDependentDialects(mlir::DialectRegistry& registry) const {
  registry.insert<StandardOpsDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::vector::VectorDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::math::MathDialect>();
}

void mlir::spn::LoSPNtoCPUStructureConversionPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalDialect<scf::SCFDialect>();
  target.addLegalDialect<mlir::math::MathDialect>();
  target.addLegalDialect<mlir::vector::VectorDialect>();
  target.addLegalDialect<mlir::memref::MemRefDialect>();
  target.addLegalOp<ModuleOp>();
  target.addLegalOp<FuncOp>();

  LoSPNtoCPUTypeConverter typeConverter;

  target.addLegalDialect<spn::low::LoSPNDialect>();
  target.addIllegalOp<spn::low::SPNKernel>();
  target.addIllegalOp<spn::low::SPNBody>();

  OwningRewritePatternList patterns(&getContext());
  spn::populateLoSPNtoCPUStructurePatterns(patterns, &getContext(), typeConverter);
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));

#define PRINT_OP_COUNTS true
#if PRINT_OP_COUNTS
  llvm::StringMap<unsigned> opCounts;
  getOperation()->walk([&](Operation* op) {
    ++opCounts[op->getName().getStringRef()];
  });
  for (auto const& entry : opCounts) {
    llvm::outs() << "op count pre conversion: \"" << entry.first() << "\": " << entry.second << "\n";
  }
#endif

  // We split this pass into two conversion pattern applications because the single task vectorization relies on
  // the structure being converted beforehand. Otherwise, the SPNBatchReads wouldn't be converted into vector loads
  // since they aren't contained in the SPNBody region like the other operations.
  if (failed(applyPartialConversion(getOperation(), target, frozenPatterns))) {
    signalPassFailure();
  }

  target.addIllegalOp<spn::low::SPNTask>();

  OwningRewritePatternList taskPatterns(&getContext());
  if (vectorize) {
    spn::populateLoSPNtoCPUVectorizationTaskPatterns(taskPatterns, &getContext(), typeConverter,
                                                     maxAttempts,
                                                     maxSuccessfulIterations,
                                                     maxNodeSize,
                                                     maxLookAhead,
                                                     reorderInstructionsDFS,
                                                     allowDuplicateElements,
                                                     allowTopologicalMixing,
                                                     useXorChains
    );
  }
  spn::populateLoSPNtoCPUTaskPatterns(taskPatterns, &getContext(), typeConverter);

  frozenPatterns = FrozenRewritePatternSet(std::move(taskPatterns));

  if (failed(applyPartialConversion(getOperation(), target, frozenPatterns))) {
    signalPassFailure();
  }

#if PRINT_OP_COUNTS
  opCounts.clear();
  getOperation()->walk([&](Operation* op) {
    ++opCounts[op->getName().getStringRef()];
  });
  for (auto const& entry : opCounts) {
    llvm::outs() << "op count post conversion: \"" << entry.first() << "\": " << entry.second << "\n";
  }
#endif
#define DO_LIVENESS_ANALYSIS true
#if DO_LIVENESS_ANALYSIS
  getOperation().walk([&](FuncOp function) {
    unsigned lifeTimeTotal = 0;
    SmallVector<double> livePercentages;
    SmallVector<Operation*> operations;
    DenseMap<Operation*, size_t> indexOf;
    {
      unsigned opIndex = 0;
      for (auto& block: function.getBlocks()) {
        for (auto& op : block) {
          operations.emplace_back(&op);
          indexOf[&op] = opIndex++;
        }
      }
    }
    for (size_t i = 0; i < operations.size(); ++i) {
      auto opIndex = indexOf[operations[i]];
      size_t lastUserIndex = opIndex;
      for (auto* user : operations[i]->getUsers()) {
        lastUserIndex = std::max(lastUserIndex, indexOf[user]);
      }
      auto isLiveFor = lastUserIndex - opIndex;
      livePercentages.emplace_back(static_cast<double>(isLiveFor) / static_cast<double>(operations.size()));
      lifeTimeTotal += isLiveFor;
    }
    double normalizedTotal = static_cast<double>(lifeTimeTotal) / static_cast<double>(operations.size());
    llvm::outs() << "lifetime total in function (" << function.getName() << "): " << normalizedTotal << "\n";
    double liveForAverage = 0;
    for (auto livePercentage : livePercentages) {
      liveForAverage += livePercentage;
    }
    liveForAverage /= indexOf.size();
    llvm::outs() << "lifetime average % in function (" << function.getName() << "): " << liveForAverage * 100 << "\n";
    double liveForStdDev = 0;
    for (auto livePercentage : livePercentages) {
      liveForStdDev += pow(livePercentage - liveForAverage, 2);
    }
    liveForStdDev = sqrt(liveForStdDev) / static_cast<double>(operations.size());
    llvm::outs() << "lifetime stddev in function (" << function.getName() << "): " << liveForStdDev * 100 << "\n";
  });
#endif

  // Useful for when we are only interested in some intermediate stats and not the compiled kernel. This reduces time
  // spent in subsequent passes practically to zero.
#define DELETE_OPS false
#if DELETE_OPS
  getOperation().walk([&](FuncOp function) {
    function.back().getTerminator()->moveBefore(&function.front().front());
    while (&function.back() != &function.front()) {
      function.getBlocks().pop_back();
    }
    while (&function.front().front() != &function.front().back()) {
      function.front().getOperations().pop_back();
    }
  });
#endif
}

void mlir::spn::LoSPNtoCPUNodeConversionPass::getDependentDialects(mlir::DialectRegistry& registry) const {
  registry.insert<StandardOpsDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<mlir::vector::VectorDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
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

std::unique_ptr<mlir::Pass> mlir::spn::createLoSPNtoCPUNodeConversionPass() {
  return std::make_unique<LoSPNtoCPUNodeConversionPass>();
}

void mlir::spn::LoSPNNodeVectorizationPass::getDependentDialects(mlir::DialectRegistry& registry) const {
  registry.insert<StandardOpsDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<mlir::vector::VectorDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
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

std::unique_ptr<mlir::Pass> mlir::spn::createLoSPNNodeVectorizationPass() {
  return std::make_unique<LoSPNNodeVectorizationPass>();
}
