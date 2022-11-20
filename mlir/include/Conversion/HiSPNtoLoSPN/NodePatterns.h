//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_HISPNTOLOSPN_JOINTNODEPATTERNS_H
#define SPNC_MLIR_INCLUDE_CONVERSION_HISPNTOLOSPN_JOINTNODEPATTERNS_H

#include "mlir/Transforms/DialectConversion.h"
#include "HiSPN/HiSPNDialect.h"
#include "HiSPN/HiSPNOps.h"
#include "llvm/Support/Debug.h"

namespace mlir {
  namespace spn {

    namespace {

      // We need the following class template with partial specialization to deconstruct
      // the variadic template step by step without ambiguous calls.

      template<typename ...Q>
      struct QueryCheck;

      template<>
      struct QueryCheck<> {
        static bool queryCheck(Operation* query) {
          return false;
        }
      };

      template<typename Q, typename ...Qs>
      struct QueryCheck<Q, Qs...> {
        static bool queryCheck(Operation* query) {
          return isa<Q>(query) || QueryCheck<Qs...>::queryCheck(query);
        }
      };

      template<typename ...Qs>
      bool checkQuery(Operation* query) {
        return QueryCheck<Qs...>::queryCheck(query);
      }

    }

    template<typename SourceOp, typename NodePattern, typename... Queries>
    struct NodeLowering : public OpConversionPattern<SourceOp> {

      using OpConversionPattern<SourceOp>::OpConversionPattern;
      using OpAdaptor = typename SourceOp::Adaptor;

      LogicalResult matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter& rewriter) const override {
        if (!checkQuery<Queries...>(op.getEnclosingQuery())) {
          return rewriter.notifyMatchFailure(op, "Enclosing query did not match");
        }
        // TODO: Messy!
        ValueRange operands = adaptor.getOperands();
        std::vector<Value> vec(operands.begin(), operands.end());
        ArrayRef<Value> _operands(vec);
        return static_cast<const NodePattern*>(this)->matchAndRewriteChecked(op, _operands, rewriter);
      }

    };

    struct ProductNodeLowering : public NodeLowering<high::ProductNode, ProductNodeLowering, high::JointQuery> {

      using NodeLowering<high::ProductNode, ProductNodeLowering, high::JointQuery>::NodeLowering;

      LogicalResult matchAndRewriteChecked(high::ProductNode op, ArrayRef<Value> operands,
                                           ConversionPatternRewriter& rewriter) const;

      Value splitProduct(high::ProductNode op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const;

    };

    struct SumNodeLowering : public NodeLowering<high::SumNode, SumNodeLowering, high::JointQuery> {

      using NodeLowering<high::SumNode, SumNodeLowering, high::JointQuery>::NodeLowering;

      LogicalResult matchAndRewriteChecked(high::SumNode op, ArrayRef<Value> operands,
                                           ConversionPatternRewriter& rewriter) const;

      Value splitWeightedSum(high::SumNode op, ArrayRef<Value> operands, ArrayRef<double> weights,
                             ConversionPatternRewriter& rewriter) const;

    };

    struct HistogramNodeLowering : public NodeLowering<high::HistogramNode, HistogramNodeLowering, high::JointQuery> {

      using NodeLowering<high::HistogramNode, HistogramNodeLowering, high::JointQuery>::NodeLowering;

      LogicalResult matchAndRewriteChecked(high::HistogramNode op, ArrayRef<Value> operands,
                                           ConversionPatternRewriter& rewriter) const;

    };

    struct CategoricalNodeLowering : public NodeLowering<high::CategoricalNode,
                                                         CategoricalNodeLowering, high::JointQuery> {

      using NodeLowering<high::CategoricalNode, CategoricalNodeLowering, high::JointQuery>::NodeLowering;

      LogicalResult matchAndRewriteChecked(high::CategoricalNode op, ArrayRef<Value> operands,
                                           ConversionPatternRewriter& rewriter) const;
    };

    struct GaussianNodeLowering : public NodeLowering<high::GaussianNode, GaussianNodeLowering, high::JointQuery> {

      using NodeLowering<high::GaussianNode, GaussianNodeLowering, high::JointQuery>::NodeLowering;

      LogicalResult matchAndRewriteChecked(high::GaussianNode op, ArrayRef<Value> operands,
                                           ConversionPatternRewriter& rewriter) const;
    };

    struct RootNodeLowering : public NodeLowering<high::RootNode, RootNodeLowering, high::JointQuery> {

      using NodeLowering<high::RootNode, RootNodeLowering, high::JointQuery>::NodeLowering;

      LogicalResult matchAndRewriteChecked(high::RootNode op, ArrayRef<Value> operands,
                                           ConversionPatternRewriter& rewriter) const;
    };

    static inline void populateHiSPNtoLoSPNNodePatterns(RewritePatternSet& patterns, MLIRContext* context,
                                                        TypeConverter& typeConverter) {
      patterns.add<ProductNodeLowering, SumNodeLowering>(typeConverter, context);
      patterns.add<HistogramNodeLowering, CategoricalNodeLowering, GaussianNodeLowering>(typeConverter, context);
      patterns.add<RootNodeLowering>(typeConverter, context);
    }

  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_HISPNTOLOSPN_JOINTNODEPATTERNS_H
