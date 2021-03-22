//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoCPU/Vectorization/SLP/SLPVectorizationPatterns.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"

using namespace mlir;
using namespace mlir::spn::low::slp;

LogicalResult VectorizeTask::matchAndRewrite(FuncOp op, PatternRewriter& rewriter) const {

  if (!op->getName().getStringRef().contains("task_")) {
    return failure();
  }
  //op->replaceUsesOfWith();

  return success();
}
VectorizeTask::VectorizeTask(MLIRContext* context, SLPGraph& graph) : OpRewritePattern(context), graph{graph} {

}
