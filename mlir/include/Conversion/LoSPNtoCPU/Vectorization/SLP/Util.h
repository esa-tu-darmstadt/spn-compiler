//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_UTIL_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_UTIL_H

#include "LoSPN/LoSPNOps.h"
#include "SLPGraph.h"
#include "mlir/IR/Operation.h"
#include <optional>

namespace mlir {
namespace spn {
namespace low {
namespace slp {

/// Returns true if the operation is vectorizable according to both the LoSPN
/// dialect's vectorizable op interface and MLIR's vector type rules.
bool vectorizable(Operation *op);
/// Returns true if the value is vectorizable according to both the LoSPN
/// dialect's vectorizable op interface and MLIR's vector type rules.
bool vectorizable(Value value);

/// Returns true if every value in [begin, end) is vectorizable.
template <typename ValueIterator>
bool vectorizable(ValueIterator begin, ValueIterator end) {
  if (begin->template isa<BlockArgument>()) {
    return false;
  }
  auto const &name = begin->getDefiningOp()->getName();
  ++begin;
  while (begin != end) {
    if (!vectorizable(*begin) || begin->getDefiningOp()->getName() != name) {
      return false;
    }
    ++begin;
  }
  return true;
}

/// Returns true if the value is vectorizable according to MLIR's vector type
/// rules.
bool ofVectorizableType(Value value);

/// Returns true if every value in [begin, end) is vectorizable according to
/// MLIR's vector type rules.
template <typename ValueIterator>
bool ofVectorizableType(ValueIterator begin, ValueIterator end) {
  return std::all_of(
      begin, end, [&](auto const &value) { return ofVectorizableType(value); });
}

/// Returns true if the value represents a commutative operation.
bool commutative(Value value);

/// Returns true if every value in [begin, end) represents a commutative
/// operation.
template <typename ValueIterator>
bool commutative(ValueIterator begin, ValueIterator end) {
  while (begin != end) {
    if (!commutative(*begin)) {
      return false;
    }
    ++begin;
  }
  return true;
}

/// Returns true if both values are loads and the second value is consecutive to
/// the first.
bool consecutiveLoads(Value lhs, Value rhs);

/// Returns true if all values in [begin, end) are loads and consecutive to
/// their predecessor.
template <typename ValueIterator>
bool consecutiveLoads(ValueIterator begin, ValueIterator end) {
  Value previous = *begin;
  if (++begin == end || previous.isa<BlockArgument>() ||
      !dyn_cast<SPNBatchRead>(previous.getDefiningOp())) {
    return false;
  }
  while (begin != end) {
    Value current = *begin;
    if (!consecutiveLoads(previous, current)) {
      return false;
    }
    previous = current;
    ++begin;
  }
  return true;
}

/// Returns true if all values in [begin, end) implement the leaf node interface
/// of the LoSPN dialect.
template <typename ValueIterator>
bool allLeaf(ValueIterator begin, ValueIterator end) {
  while (begin != end) {
    if (auto *definingOp = begin->getDefiningOp()) {
      if (!dyn_cast<LeafNodeInterface>(definingOp)) {
        return false;
      }
      ++begin;
    } else {
      return false;
    }
  }
  return true;
}

/// Return true if the superword contains any marginalized normal distributions.
bool anyGaussianMarginalized(Superword const &superword);

/// Return the operands of the value as a C++ vector.
SmallVector<Value, 2> getOperands(Value value);

/// Sort all provided values in place by their opcode (i.e. in lexicographical
/// order). An optional smallest opcode can be provided to force certain values
/// to the front.
void sortByOpcode(SmallVectorImpl<Value> &values,
                  std::optional<OperationName> smallestOpcode = std::nullopt);

/// Print a textual representation of the superword to the console.
void dumpSuperword(Superword const &superword);
/// Print a textual representation of the SLP node to the console.
void dumpSLPNode(SLPNode const &node);

/// Print a dot-language representation of the operand trees of the provided
/// values to the console.
void dumpOpGraph(ArrayRef<Value> values);
/// Print a dot-language representation of the superword tree with the provided
/// root to the console.
void dumpSuperwordGraph(Superword *root);
/// Print a dot-language representation of the SLP graph with the provided root
/// node to the console.
void dumpSLPGraph(SLPNode *root, bool includeInputs = false);
/// Print a dot-language representation of the dependency graph to the console.
void dumpDependencyGraph(DependencyGraph const &dependencyGraph);

} // namespace slp
} // namespace low
} // namespace spn
} // namespace mlir

#endif // SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_UTIL_H
