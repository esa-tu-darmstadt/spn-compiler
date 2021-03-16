//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <mlir/IR/Function.h>
#include "mlir/IR/Module.h"
#include "VectorizationPatterns.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"

mlir::LogicalResult mlir::spn::slp::GaussianOpVectorization::matchAndRewrite(mlir::spn::GaussianOp op,
                                                                             mlir::PatternRewriter& rewriter) const {

  return failure();

  if (!isAssignedToVector(op.getOperation())) {
    return failure();
  }

  auto a = op.getOperation();

  for (auto const& index : vectorIndices.at(op.getOperation())) {
    auto const& vector = vectors[index];

    if (!isVectorMixed(vector)) {
      auto const& n = vector.size();
      Value values[n];
      Value indices[n];
      double means[n];
      double stddevs[n];

      for (size_t i = 0; i < n; ++i) {
        indices[i] = dyn_cast<GaussianOp>(vector[i]).index();
        means[i] = dyn_cast<GaussianOp>(vector[i]).mean().convertToDouble();
        stddevs[i] = dyn_cast<GaussianOp>(vector[i]).stddev().convertToDouble();
        indices[i].dump();
        values[i] = dyn_cast<GaussianOp>(vector[i]).getOperand();
      }
/*
      auto t = rewriter.create<GaussianVectorOp>(op.getLoc(),
                                        op.index(),
                                        llvm::ArrayRef<double>(means, n),
                                        llvm::ArrayRef<double>(stddevs, n));
      ValueRange valueRange{};
      rewriter.replaceOp(t.getOperation(), {});
      t.getOperation()->dump();
      auto x = dyn_cast<GaussianVectorOp>(t.getOperation()).getFeatureIndex();

      x = x - 1;*/
    } else {

    }
    for (auto const& gaussianOp : vector) {
      rewriter.eraseOp(gaussianOp);
    }
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

mlir::LogicalResult mlir::spn::slp::SumOpVectorization::matchAndRewrite(mlir::spn::SumOp op,
                                                                        mlir::PatternRewriter& rewriter) const {

  if (!isAssignedToVector(op.getOperation())) {
    return failure();
  }

  auto a = op.getOperation();

  for (auto const& index : vectorIndices.at(op.getOperation())) {
    auto const& vector = vectors[index];
    Type vectorType = VectorType::get(vector.size(), op.getType());
    Type memrefType = MemRefType::get(vector.size(), op.getType());
    ValueRange range;
    rewriter.create<AllocaOp>(op.getLoc(), memrefType, range);
    op.getOperation()->getBlock()->dump();
  }
  /*
  if (isVectorMixed(vector)) {
    OpBuilder builder(op.getContext());
    builder.setInsertionPoint(op);
    llvm::SmallVector<Value, 4> values;
    values.reserve(vector.size());
    for (size_t i = 0; i < vector.size(); ++i) {
     values[i] = vector[i]->getResult(0);
    }
    //DenseElementsAttr::get(vectorType, llvm::ArrayRef<Value>{values});
  }

  for (size_t i = 0; i < n; ++i) {
    indices[i] = dyn_cast<GaussianOp>(vector[i]).index();
    means[i] = dyn_cast<GaussianOp>(vector[i]).mean().convertToDouble();
    stddevs[i] = dyn_cast<GaussianOp>(vector[i]).stddev().convertToDouble();
    indices[i].dump();
    values[i] = dyn_cast<GaussianOp>(vector[i]).getOperand();
  }

    auto t = rewriter.create<vector::LoadOp>(op.getLoc(),
                                      op.index(),
                                      llvm::ArrayRef<double>(means, n),
                                      llvm::ArrayRef<double>(stddevs, n));
    ValueRange valueRange{};
    rewriter.replaceOp(t.getOperation(), {});
    t.getOperation()->dump();
    for (size_t i = 0; i < vector.size(); ++i) {
     values[i] = vector[i]->getResult(0);
    }
    //DenseElementsAttr::get(vectorType, llvm::ArrayRef<Value>{values});
  }

  for (size_t i = 0; i < n; ++i) {
    indices[i] = dyn_cast<SumOp>(vector[i]).index();
    means[i] = dyn_cast<SumOp>(vector[i]).mean().convertToDouble();
    stddevs[i] = dyn_cast<SumOp>(vector[i]).stddev().convertToDouble();
    indices[i].dump();
    values[i] = dyn_cast<SumOp>(vector[i]).getOperand();
  }

    x = x - 1;
} else {

}
for (auto const& gaussianOp : vector) {
  rewriter.eraseOp(gaussianOp);
}
}

}
   */
  return success();
}