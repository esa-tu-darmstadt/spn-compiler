//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPATTERNS_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPATTERNS_H

#include "mlir/Transforms/DialectConversion.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPGraph.h"

namespace mlir {
  namespace spn {

    struct VectorizeTask : OpRewritePattern<FuncOp> {

      using OpRewritePattern<FuncOp>::OpRewritePattern;

      explicit VectorizeTask(MLIRContext* context, SLPGraph& graph);

      LogicalResult matchAndRewrite(FuncOp op, PatternRewriter& rewriter) const override;

    private:
      SLPGraph& graph;
    };
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPVECTORIZATIONPATTERNS_H
