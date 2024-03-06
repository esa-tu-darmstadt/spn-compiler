//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_GRAPHCONVERSION_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_GRAPHCONVERSION_H

#include "PatternVisitors.h"
#include "SLPGraph.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/FoldUtils.h"

namespace mlir {
namespace spn {
namespace low {
namespace slp {

class CostModel;

/// A ValuePosition models the position of a value in a superword, i.e. in which
/// superword and at which position a value is located.
struct ValuePosition {
  ValuePosition() = default;
  ValuePosition(Superword *superword, size_t index)
      : superword{superword}, index{index} {}
  explicit operator bool() const { return superword; }
  Superword *superword;
  size_t index;
};

/// The ConversionState keeps track of the state of vectorization in terms of
/// which value/superword has been marked as computed (or extracted) already. It
/// also stores the SLP graphs for subsequent vectorizations.
class ConversionState {
  friend class ConversionManager;

public:
  /// Check if the superword has been (marked as) computed already.
  bool alreadyComputed(Superword *superword) const;
  /// Check if the value has been (marked as) computed already.
  bool alreadyComputed(Value value) const;
  /// Check if the value is extractable, i.e. there exists a superword that has
  /// been (marked as) computed which also contains that value.
  bool isExtractable(Value value);

  /// Temporarily mark a value as computed (e.g. if required for inputs when
  /// packing vectors). Once the conversion is finished, it will automatically
  /// be marked permanently or dismissed depending on the profitability of the
  /// vectorization. Also recursively marks the value's operands as computed.
  void markComputed(Value value);
  /// Temporarily mark a superword as computed. Once the conversion is finished,
  /// it will automatically be marked permanently or dismissed depending on the
  /// profitability of the vectorization. The provided value should be the value
  /// of the last operation that is required for the computation of the
  /// superword.
  void markComputed(Superword *superword, Value value);
  /// Temporarily mark a value as extracted. Once the conversion is finished, it
  /// will automatically be marked permanently or dismissed depending on the
  /// profitability of the vectorization.
  void markExtracted(Value value);

  Value getValue(Superword *superword) const;
  ValuePosition getSuperwordContainingValue(Value value) const;

  // Callback registration.
  /// Callbacks for when a superword was converted and for when its value has
  /// been removed because the graph it was contained in was not deemed
  /// profitable.
  void addVectorCallbacks(std::function<void(Superword *)> createCallback,
                          std::function<void(Superword *)> undoCallback);
  /// Callbacks for when a scalar value is being used as input for some vector
  /// and for when a scalar that was previously used as input for some vector is
  /// no longer an input because the corresponding graph was not deemed
  /// profitable.
  void addScalarCallbacks(std::function<void(Value)> inputCallback,
                          std::function<void(Value)> undoCallback);
  /// Callbacks for when an extraction for some value has been created and for
  /// when an extraction for some value has been undone because the
  /// corresponding graph was not deemed profitable.
  void addExtractionCallbacks(std::function<void(Value)> extractCallback,
                              std::function<void(Value)> undoCallback);

private:
  /// Notify the conversion state that the conversion of a new SLP graph will
  /// take place.
  void startConversion(std::shared_ptr<Superword> root);
  /// Notify the conversion state that the conversion of the current SLP graph
  /// has both been finished and deemed profitable. Every superword/value that
  /// has temporarily been marked as computed will be marked permanently.
  void finishConversion();
  /// Notify the conversion state that the conversion of the current SLP graph
  /// has been finished but not deemed profitable. Every superword/value that
  /// has temporarily been marked as computed will not be marked as computed
  /// anymore.
  void cancelConversion();
  // Take ownership of the graphs to prevent dangling superword pointers when
  // the graphs go out of scope.
  SmallVector<std::shared_ptr<Superword>, 5> correspondingGraphs;

  /// For bookkeeping of computed superwords and values.
  struct ConversionData {

    bool alreadyComputed(Value value) const {
      return computedScalarValues.contains(value) ||
             extractedScalarValues.contains(value);
    }

    void clear() {
      computedScalarValues.clear();
      computedSuperwords.clear();
      extractedScalarValues.clear();
      extractableScalarValues.clear();
    }

    void mergeWith(ConversionData &other) {
      computedScalarValues.insert(std::begin(other.computedScalarValues),
                                  std::end(other.computedScalarValues));
      computedSuperwords.copyFrom(other.computedSuperwords);
      extractedScalarValues.insert(std::begin(other.extractedScalarValues),
                                   std::end(other.extractedScalarValues));
      extractableScalarValues.copyFrom(other.extractableScalarValues);
    }
    /// Scalar values that are marked as computed (e.g. because they're used as
    /// inputs for vectors).
    SmallPtrSet<Value, 32> computedScalarValues;
    /// Superwords that are marked as computed.
    DenseMap<Superword *, Value> computedSuperwords;
    /// Extractions that have taken place.
    SmallPtrSet<Value, 32> extractedScalarValues;
    /// Store vector element data for faster extraction lookup.
    DenseMap<Value, ValuePosition> extractableScalarValues;
  };

  /// Permanent conversion data.
  ConversionData permanentData;
  /// Temporary conversion data. These might need to be 'undone' when a graph is
  /// not deemed profitable.
  ConversionData temporaryData;
  // Callback data.
  SmallVector<std::function<void(Superword *)>> vectorCallbacks;
  SmallVector<std::function<void(Superword *)>> vectorUndoCallbacks;
  SmallVector<std::function<void(Value)>> scalarCallbacks;
  SmallVector<std::function<void(Value)>> scalarUndoCallbacks;
  SmallVector<std::function<void(Value)>> extractionCallbacks;
  SmallVector<std::function<void(Value)>> extractionUndoCallbacks;
};

/// The ConversionManager handles everything that's related to converting SLP
/// graphs back into SIMD instructions. This includes things like keeping track
/// of superword -> SIMD operation mappings, creating constants and setting
/// insertion points.
class ConversionManager {

public:
  /// The rewriter is required for creating constant operations and extractions
  /// (for escaping uses). All created operations will be inserted into the
  /// provided block. The cost model is required for determining the
  /// profitability of extractions.
  ConversionManager(RewriterBase &rewriter, Block *block, CostModel *costModel,
                    bool reorderInstructionsDFS);

  /// Begin the conversion of a new SLP graph. Also computes escaping uses for
  /// later on during the conversion process. Returns the order in which the
  /// graph's superwords should be converted.
  SmallVector<Superword *> startConversion(SLPGraph const &graph);
  /// Method to call to conclude the conversion process of an SLP graph. Should
  /// be called if the vectorization was deemed profitable as it permanently
  /// keeps all changes done to the original basic block. Also reorders
  /// operations such that no operation appears before its operands.
  void finishConversion();
  /// Method to call to cancel the conversion process of an SLP graph. Should be
  /// called if the vectorization was not deemed profitable. Removes all
  /// operations that were created by the conversion and restores the original
  /// state of the basic block.
  void cancelConversion();

  /// Sets up the conversion for a single superword by setting an appropriate
  /// insertion point. Also creates extractions for any needed scalar values
  /// (depending on the provided pattern) if these are more profitable than
  /// computing them in a scalar fashion.
  void setupConversionFor(Superword *superword,
                          SLPVectorizationPattern const *pattern);
  /// Concludes the conversion a single superword. The provided value is used to
  /// internally map that superword to its actual SIMD operation, so it should
  /// be the last operation needed to 'compute' the superword. The pattern is
  /// required to mark any required scalar values as computed. Also creates
  /// extractions for escaping users.
  void update(Superword *superword, Value operation,
              SLPVectorizationPattern const *appliedPattern);

  /// Return the SIMD operation that was created for the provided superword. If
  /// more than one was created, returns the last one.
  Value getValue(Superword *superword) const;
  /// Returns a constant operation for the provided attribute. If there is no
  /// such constant operation already, creates one at the provided location
  /// automatically.
  Value getOrCreateConstant(Location const &loc, Attribute const &attribute);
  /// Return the internally stored conversion state, e.g. for callback
  /// registration.
  ConversionState &getConversionState() const;

private:
  bool hasEscapingUsers(Value value) const;
  Value getOrExtractValue(Value value);
  void reorderOperations();

  Block *block;
  CostModel *costModel;
  std::shared_ptr<ConversionState> conversionState;

  /// Stores escaping users for each value.
  DenseMap<Value, SmallVector<Operation *, 1>> escapingUsers;

  /// For reverting back to pre-conversion block states.
  SmallVector<Operation *, 32> originalOperations;
  DenseMap<Operation *, SmallVector<Value, 2>> originalOperands;

  /// Helps find out which vector elements can be erased.
  LeafPatternVisitor leafVisitor;

  /// For creating constants, setting insertion points, creating extractions,
  /// ....
  RewriterBase &rewriter;

  /// Helps avoid creating duplicate constants.
  OperationFolder folder;

  // === SLP options === //
  bool reorderInstructionsDFS;
};
} // namespace slp
} // namespace low
} // namespace spn
} // namespace mlir

#endif // SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_GRAPHCONVERSION_H
