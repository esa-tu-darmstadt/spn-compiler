//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPU/Vectorization/SLP/SLPVectorizationPass.h"
#include "LoSPNtoCPU/Vectorization/TargetInformation.h"
#include "LoSPNtoCPU/LoSPNtoCPUTypeConverter.h"
#include "LoSPNtoCPU/StructurePatterns.h"
#include "LoSPNtoCPU/NodePatterns.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPSeeding.h"

void mlir::spn::SLPVectorizationPass::runOnOperation() {
  llvm::StringRef funcName = getOperation().getName();
  if (!funcName.contains("task_")) {
    return;
  }

  ConversionTarget target(getContext());

  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalDialect<mlir::scf::SCFDialect>();
  target.addLegalDialect<mlir::vector::VectorDialect>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
  target.addLegalOp<FuncOp>();

  target.addLegalDialect<mlir::spn::low::LoSPNDialect>();

  auto function = getOperation();

  auto& seedAnalysis = getAnalysis<mlir::spn::slp::SeedAnalysis>();

  auto computationType = function.getArguments().back().getType().dyn_cast<MemRefType>().getElementType();
  auto width = TargetInformation::nativeCPUTarget().getHWVectorEntries(computationType);

  getOperation()->dump();

  auto seeds = seedAnalysis.getSeeds(width, seedAnalysis.getOpDepths());
  llvm::outs() << "nnn\n";
/*
  if (!seeds.empty()) {
    slp::SLPTree graph(seeds.front(), 3);
    OwningRewritePatternList patterns;
    std::vector<std::vector<Operation*>> vectors;
    for (auto const& node_ptr : graph.getNodes()) {
      for (size_t i = 0; i < node_ptr->numVectors(); ++i) {
        vectors.emplace_back(node_ptr->getVector(i));
      }
    }
    slp::populateVectorizationPatterns(patterns, &getContext(), vectors);
    auto op = getOperation();
    FrozenRewritePatternList frozenPatterns(std::move(patterns));
    applyPatternsAndFoldGreedily(op, frozenPatterns);
  }
*/


}

std::unique_ptr<mlir::Pass> mlir::spn::createSLPVectorizationPass() {
  return std::make_unique<SLPVectorizationPass>();
}

