//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPN/LoSPNDialect.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPVectorizationPass.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPUtil.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPGraphBuilder.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPVectorizationPatterns.h"
#include "LoSPNtoCPU/Vectorization/TargetInformation.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"

using namespace mlir::spn::low::slp;

void SLPVectorizationPass::runOnOperation() {
  llvm::StringRef funcName = getOperation().getName();
  if (!funcName.contains("task_")) {
    return;
  }

  auto function = getOperation();

  auto& seedAnalysis = getAnalysis<SeedAnalysis>();

  auto computationType = function.getArguments().back().getType().dyn_cast<MemRefType>().getElementType();
  auto width = TargetInformation::nativeCPUTarget().getHWVectorEntries(computationType);

  auto const& seed = seedAnalysis.getSeed(width, SearchMode::UseBeforeDef);
  assert(!seed.empty() && "couldn't find a seed!");
  // dumpOpTree(seed);
  SLPGraphBuilder builder{3};
  auto graph = builder.build(seed);
  // dumpSLPGraph(*graph);
  // ==== //

  SLPVectorPatternRewriter rewriter(&getContext());
  if (failed(rewriter.rewrite(graph.get()))) {
    signalPassFailure();
  }

  getOperation()->dump();

}

std::unique_ptr<mlir::Pass> mlir::spn::createSLPVectorizationPass() {
  return std::make_unique<SLPVectorizationPass>();
}
