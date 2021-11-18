//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNtoCPU/Vectorization/SLP/PatternVisitors.h"

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::low;
using namespace mlir::spn::low::slp;


// === PatternVisitor === //

void PatternVisitor::visit(BroadcastSuperword const* pattern, Superword const* superword) {
  visitDefault(pattern, superword);
}

void PatternVisitor::visit(BroadcastInsertSuperword const* pattern, Superword const* superword) {
  visitDefault(pattern, superword);
}

void PatternVisitor::visit(ShuffleTwoSuperwords const* pattern, Superword const* superword) {
  visitDefault(pattern, superword);
}

void PatternVisitor::visit(VectorizeConstant const* pattern, Superword const* superword) {
  visitDefault(pattern, superword);
}

void PatternVisitor::visit(CreateConsecutiveLoad const* pattern, Superword const* superword) {
  visitDefault(pattern, superword);
}

void PatternVisitor::visit(CreateGatherLoad const* pattern, Superword const* superword) {
  visitDefault(pattern, superword);
}

void PatternVisitor::visit(VectorizeAdd const* pattern, Superword const* superword) {
  visitDefault(pattern, superword);
}

void PatternVisitor::visit(VectorizeMul const* pattern, Superword const* superword) {
  visitDefault(pattern, superword);
}

void PatternVisitor::visit(VectorizeGaussian const* pattern, Superword const* superword) {
  visitDefault(pattern, superword);
}

void PatternVisitor::visit(VectorizeLogAdd const* pattern, Superword const* superword) {
  visitDefault(pattern, superword);
}

void PatternVisitor::visit(VectorizeLogMul const* pattern, Superword const* superword) {
  visitDefault(pattern, superword);
}

void PatternVisitor::visit(VectorizeLogGaussian const* pattern, Superword const* superword) {
  visitDefault(pattern, superword);
}

// === LeafPatternVisitor === //

ArrayRef<Value> LeafPatternVisitor::getRequiredScalarValues(SLPVectorizationPattern const* pattern,
                                                            Superword const* superword) {
  pattern->accept(*this, superword);
  return this->scalarValues;
}

void LeafPatternVisitor::visitDefault(SLPVectorizationPattern const* pattern, Superword const* superword) {
  this->scalarValues.clear();
}

void LeafPatternVisitor::visit(BroadcastSuperword const* pattern, Superword const* superword) {
  this->scalarValues.assign({superword->getElement(0)});
}

void LeafPatternVisitor::visit(BroadcastInsertSuperword const* pattern, Superword const* superword) {
  SmallPtrSet<Value, 4> uniqueValues{std::begin(*superword), std::end(*superword)};
  this->scalarValues.assign(std::begin(uniqueValues), std::end(uniqueValues));
}
