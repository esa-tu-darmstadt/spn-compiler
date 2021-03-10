//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <mlir/IR/Function.h>
#include "mlir/IR/Module.h"
#include "VectorizationPatterns.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Dialect/Vector/VectorOps.h"

mlir::LogicalResult mlir::spn::slp::GaussianOpVectorization::matchAndRewrite(mlir::spn::GaussianOp op,
                                                                             mlir::PatternRewriter& rewriter) const {

  if (!assignedToVector(op.getOperation())) {
    return failure();
  }

  // llvm::ArrayRef<double> means, llvm::ArrayRef<double> stddevs
  auto const& vector = vectors[3];
  auto const& n = vector.size();
  Value indices[n];
  double means[n];
  double stddevs[n];

  for (size_t i = 0; i < n; ++i) {
    indices[i] = dyn_cast<GaussianOp>(vector[i]).index();
    means[i] = dyn_cast<GaussianOp>(vector[i]).mean().convertToDouble();
    stddevs[i] = dyn_cast<GaussianOp>(vector[i]).stddev().convertToDouble();
  }

  rewriter.create<GaussianVectorOp>(op.getLoc(),
                                    op.index(),
                                    llvm::ArrayRef<double>(means, n),
                                    llvm::ArrayRef<double>(stddevs, n));

  for (auto const& gaussianOp : vector) {
    rewriter.eraseOp(gaussianOp);
  }

  if (false) {
    if (!op.mean().isExactlyValue(-7.0)) {
      rewriter.replaceOpWithNewOp<mlir::spn::GaussianOp>(op,
                                                         op.getType(),
                                                         op.index(), llvm::APFloat(-7.0)/*op.mean()*/,
                                                         op.stddev());

    }
  }

  return success();
}