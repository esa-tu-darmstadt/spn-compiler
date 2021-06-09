//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPNtoCPU/Vectorization/SLP/PatternVisitors.h"
#include "LoSPNtoCPU/Vectorization/SLP/SLPVectorizationPatterns.h"

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::low;
using namespace mlir::spn::low::slp;


// === PatternVisitor === //

void PatternVisitor::visit(BroadcastSuperword* pattern, Superword* superword) {
  visitDefault(pattern, superword);
}

void PatternVisitor::visit(BroadcastInsertSuperword* pattern, Superword* superword) {
  visitDefault(pattern, superword);
}

void PatternVisitor::visit(VectorizeConstant* pattern, Superword* superword) {
  visitDefault(pattern, superword);
}

void PatternVisitor::visit(VectorizeBatchRead* pattern, Superword* superword) {
  visitDefault(pattern, superword);
}

void PatternVisitor::visit(VectorizeAdd* pattern, Superword* superword) {
  visitDefault(pattern, superword);
}

void PatternVisitor::visit(VectorizeMul* pattern, Superword* superword) {
  visitDefault(pattern, superword);
}

void PatternVisitor::visit(VectorizeGaussian* pattern, Superword* superword) {
  visitDefault(pattern, superword);
}

void PatternVisitor::visit(VectorizeLogAdd* pattern, Superword* superword) {
  visitDefault(pattern, superword);
}

void PatternVisitor::visit(VectorizeLogMul* pattern, Superword* superword) {
  visitDefault(pattern, superword);
}

void PatternVisitor::visit(VectorizeLogGaussian* pattern, Superword* superword) {
  visitDefault(pattern, superword);
}

// === ScalarValueVisitor === //

ArrayRef<Value> ScalarValueVisitor::getRequiredScalarValues(SLPVectorizationPattern* pattern, Superword* superword) {
  pattern->accept(*this, superword);
  return this->scalarValues;
}

void ScalarValueVisitor::visitDefault(SLPVectorizationPattern* pattern, Superword* superword) {
  this->scalarValues.clear();
}

void ScalarValueVisitor::visit(BroadcastSuperword* pattern, Superword* superword) {
  this->scalarValues.assign({superword->getElement(0)});
}

void ScalarValueVisitor::visit(BroadcastInsertSuperword* pattern, Superword* superword) {
  SmallPtrSet<Value, 4> uniqueValues{std::begin(*superword), std::end(*superword)};
  this->scalarValues.assign(std::begin(uniqueValues), std::end(uniqueValues));
}

// === LeafPatternVisitor === //

bool LeafPatternVisitor::isLeafPattern(SLPVectorizationPattern* pattern) {
  pattern->accept(*this, nullptr);
  return this->isLeaf;
}

void LeafPatternVisitor::visitDefault(SLPVectorizationPattern* pattern, Superword* superword) {
  this->isLeaf = false;
}

void LeafPatternVisitor::visit(BroadcastSuperword* pattern, Superword* superword) {
  this->isLeaf = true;
}

void LeafPatternVisitor::visit(BroadcastInsertSuperword* pattern, Superword* superword) {
  this->isLeaf = true;
}

void LeafPatternVisitor::visit(VectorizeConstant* pattern, Superword* superword) {
  this->isLeaf = true;
}

void LeafPatternVisitor::visit(VectorizeBatchRead* pattern, Superword* superword) {
  this->isLeaf = true;
}
