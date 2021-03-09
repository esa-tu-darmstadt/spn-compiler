//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <mlir/IR/Function.h>
#include "mlir/IR/Module.h"
#include "VectorPatterns.h"
#include "mlir/IR/BlockAndValueMapping.h"

mlir::LogicalResult mlir::spn::slp::GaussianOpVectorization::matchAndRewrite(mlir::spn::GaussianOp op,
                                                                             mlir::PatternRewriter& rewriter) const {
  if (!op.mean().isExactlyValue(-7.0)) {
    rewriter.replaceOpWithNewOp<mlir::spn::GaussianOp>(op,
                                                       op.getType(),
                                                       op.index(),
                                                       llvm::APFloat(-7.0), /*op.mean(),*/
                                                       op.stddev());
  }
  return success();
}