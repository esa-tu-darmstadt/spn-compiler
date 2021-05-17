//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_LIB_DIALECT_LOSPN_PARTITIONING_GRAPHPARTITIONER_H
#define SPNC_MLIR_LIB_DIALECT_LOSPN_PARTITIONING_GRAPHPARTITIONER_H

#include "LoSPN/LoSPNOps.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {
  namespace spn {
    namespace low {

      class Partition {

      public:

        void addNode(Operation* node);

        void removeNode(Operation* node);

        bool contains(Operation* node);

        SmallPtrSetImpl<Operation*>::const_iterator begin();

        SmallPtrSetImpl<Operation*>::const_iterator end();

        llvm::ArrayRef<Operation*> hasExternalInputs();

        llvm::ArrayRef<Operation*> hasExternalOutputs();

      private:

        llvm::SmallPtrSet<Operation*, 32> nodes;

        bool dirty = false;

        llvm::SmallVector<Operation*> extIn;

        llvm::SmallVector<Operation*> exOut;

        void computeExternalConnections();

      };

      class GraphPartitioner {

      public:

        // TODO
        explicit GraphPartitioner() = default;

        SmallVector<std::unique_ptr<Partition>> partitionGraph(llvm::ArrayRef<Operation*> nodes,
                                                               llvm::ArrayRef<Operation*> inNodes,
                                                               llvm::ArrayRef<Value> externalInputs);

      private:

        SmallVector<std::unique_ptr<Partition>> initialPartitioning(llvm::ArrayRef<Operation*> nodes,
                                                                    llvm::ArrayRef<Operation*> inNodes,
                                                                    llvm::ArrayRef<Value> externalInputs) const;

        bool hasInDegreeZero(Operation* node, llvm::SmallPtrSetImpl<Operation*>& partitioned,
                             llvm::SmallPtrSetImpl<Value>& externalInputs) const;

      };

    }
  }
}

#endif //SPNC_MLIR_LIB_DIALECT_LOSPN_PARTITIONING_GRAPHPARTITIONER_H
