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
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/BitVector.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        class SeedAnalysis {

        public:
          SeedAnalysis(Operation* rootOp, unsigned width);
          virtual ~SeedAnalysis() = default;
          virtual SmallVector<Value, 4> next();
          void update(ArrayRef<Superword*> convertedSuperwords);
        protected:
          virtual void computeAvailableOps() = 0;
          virtual SmallVector<Value, 4> nextSeed() const = 0;
          Operation* const rootOp;
          unsigned const width;
          // SetVector to make seeding deterministic from run to run.
          llvm::SmallSetVector<Operation*, 32> availableOps;
        private:
          bool availableComputed = false;
        };

        class TopDownAnalysis : public SeedAnalysis {
        public:
          TopDownAnalysis(Operation* rootOp, unsigned width);
          SmallVector<Value, 4> nextSeed() const override;
        protected:
          void computeAvailableOps() override;
        };

        /// Deprecated (worse runtime compared to topdown for same results).
        class FirstRootAnalysis : public SeedAnalysis {
        public:
          FirstRootAnalysis(Operation* rootOp, unsigned width);
          SmallVector<Value, 4> nextSeed() const override;
        protected:
          void computeAvailableOps() override;
        private:
          Operation* findFirstRoot(llvm::StringMap<DenseMap<Operation*, llvm::BitVector>>& reachableLeaves) const;
        };

      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SEEDING_H
