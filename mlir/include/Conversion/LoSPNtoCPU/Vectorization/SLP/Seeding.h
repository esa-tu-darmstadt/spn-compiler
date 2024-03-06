//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SEEDING_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SEEDING_H

#include "SLPGraph.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
namespace spn {
namespace low {
namespace slp {

/// A seed analysis is responsible for finding seed instructions for SLP
/// vectorization in a program. Every analysis computes a set of available
/// operations once and then continuously updates the set from vectorization
/// iteration to vectorization iteration.
class SeedAnalysis {

public:
  SeedAnalysis(Operation *rootOp, unsigned width);
  virtual ~SeedAnalysis() = default;
  /// Get the next seed.
  virtual SmallVector<Value, 4> next();
  /// Notify the analysis that an SLP graph consisting of the provided
  /// superwords has successfully been converted. The operations contained in
  /// the superwords should not appear in subsequent seeds.
  void update(ArrayRef<Superword *> convertedSuperwords);

protected:
  virtual void computeAvailableOps() = 0;
  virtual SmallVector<Value, 4> nextSeed() const = 0;
  Operation *const rootOp;
  unsigned const width;
  // SetVector to make seeding deterministic from run to run.
  llvm::SmallSetVector<Operation *, 32> availableOps;

private:
  bool availableComputed = false;
};

/// A seeding analysis that computes the seed in the program in a top-down
/// manner, i.e. from the program terminator towards the program inputs.
class TopDownAnalysis : public SeedAnalysis {
public:
  TopDownAnalysis(Operation *rootOp, unsigned width);

protected:
  void computeAvailableOps() override;
  SmallVector<Value, 4> nextSeed() const override;
};

/// A seeding analysis that computes the seed in a bottom-up fashion, i.e. it
/// starts at the program inputs, tries to find a root that covers all inputs
/// and uses that root to retrieve a seed in a top-down manner instead of the
/// program terminator. \deprecated worse runtime compared to topdown for same
/// results
class FirstRootAnalysis : public SeedAnalysis {
public:
  FirstRootAnalysis(Operation *rootOp, unsigned width);

protected:
  void computeAvailableOps() override;
  SmallVector<Value, 4> nextSeed() const override;

private:
  /// Retrieves the first common root of the provided leaf operations, or
  /// nullptr if there is none that is available anymore. Fills the map of
  /// reachable leaves for every operation during the process, i.e. maps
  /// operations to those leaves that 'flow into' the operation (1 if a leaf
  /// does, 0 otherwise).
  Operation *findRoot(SmallPtrSet<Operation *, 32> const &leaves,
                      llvm::StringMap<DenseMap<Operation *, llvm::BitVector>>
                          &reachableLeaves) const;
};

} // namespace slp
} // namespace low
} // namespace spn
} // namespace mlir

#endif // SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SEEDING_H
