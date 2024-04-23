//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_PATTERNVISITORS_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_PATTERNVISITORS_H

#include "SLPGraph.h"
#include "SLPVectorizationPatterns.h"

namespace mlir {
namespace spn {
namespace low {
namespace slp {

/// A PatternVisitor can be used to compute information for a superword,
/// depending on the pattern to be applied (e.g. any scalar values a superword
/// requires for every pattern). Every visitor should store computed data
/// internally as no values are returned, i.e. some method should be provided
/// for accessing the information.
class PatternVisitor {
public:
  /// Visit a superword under the BroadcastSuperword pattern.
  virtual void visit(BroadcastSuperword const *pattern,
                     Superword const *superword);
  /// Visit a superword under the BroadcastInsertSuperword pattern.
  virtual void visit(BroadcastInsertSuperword const *pattern,
                     Superword const *superword);
  /// Visit a superword under the ShuffleTwoSuperwords pattern.
  virtual void visit(ShuffleTwoSuperwords const *pattern,
                     Superword const *superword);
  /// Visit a superword under the VectorizeConstant pattern.
  virtual void visit(VectorizeConstant const *pattern,
                     Superword const *superword);
  /// Visit a superword under the CreateConsecutiveLoad pattern.
  virtual void visit(CreateConsecutiveLoad const *pattern,
                     Superword const *superword);
  /// Visit a superword under the CreateGatherLoad pattern.
  virtual void visit(CreateGatherLoad const *pattern,
                     Superword const *superword);
  /// Visit a superword under the VectorizeAdd pattern.
  virtual void visit(VectorizeAdd const *pattern, Superword const *superword);
  /// Visit a superword under the VectorizeMul pattern.
  virtual void visit(VectorizeMul const *pattern, Superword const *superword);
  /// Visit a superword under the VectorizeGaussian pattern.
  virtual void visit(VectorizeGaussian const *pattern,
                     Superword const *superword);
  /// Visit a superword under the VectorizeLogAdd pattern.
  virtual void visit(VectorizeLogAdd const *pattern,
                     Superword const *superword);
  /// Visit a superword under the VectorizeLogMul pattern.
  virtual void visit(VectorizeLogMul const *pattern,
                     Superword const *superword);
  /// Visit a superword under the VectorizeLogGaussian pattern.
  virtual void visit(VectorizeLogGaussian const *pattern,
                     Superword const *superword);

protected:
  virtual ~PatternVisitor() = default;
  /// Default visiting method so that we don't have to override every single
  /// visit method in each visitor.
  virtual void visitDefault(SLPVectorizationPattern const *pattern,
                            Superword const *superword) = 0;
};

/// A LeafPatternVisitor visits leaf superwords of an SLP graph and computes
/// which of their elements need to be computed/available in a scalar fashion
/// beforehand.
class LeafPatternVisitor : public PatternVisitor {
public:
  /// Returns the scalar values whose scalar computation is required for the
  /// superword if the provided pattern were to be applied.
  ArrayRef<Value>
  getRequiredScalarValues(SLPVectorizationPattern const *pattern,
                          Superword const *superword);
  void visit(BroadcastSuperword const *pattern,
             Superword const *superword) override;
  void visit(BroadcastInsertSuperword const *pattern,
             Superword const *superword) override;

protected:
  void visitDefault(SLPVectorizationPattern const *pattern,
                    Superword const *superword) override;

private:
  SmallVector<Value, 4> scalarValues;
};

} // namespace slp
} // namespace low
} // namespace spn
} // namespace mlir

#endif // SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_PATTERNVISITORS_H
