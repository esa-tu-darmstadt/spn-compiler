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
#include "LoSPNtoCPU/Vectorization/SLP/SLPGraph.h"

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

  auto seeds = seedAnalysis.getSeeds(width, seedAnalysis.getOpDepths(), slp::UseBeforeDef);
  assert(!seeds.empty() && "couldn't find a seed!");

  slp::SLPGraph graph(seeds.front(), 3);
  transform(graph);

}

void mlir::spn::SLPVectorizationPass::transform(slp::SLPGraph& graph) {
  graph.dump();
  transform(graph.getRoot(), true);
}

mlir::Operation* mlir::spn::SLPVectorizationPass::transform(slp::SLPNode& node, bool isRoot) {
  bool isMixed = false;
  return node.getVector(0).front();
}

std::unique_ptr<mlir::Pass> mlir::spn::createSLPVectorizationPass() {
  return std::make_unique<SLPVectorizationPass>();
}

