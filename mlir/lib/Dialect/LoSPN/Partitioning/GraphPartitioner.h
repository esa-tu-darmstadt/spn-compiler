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
#include "Heuristic.h"

namespace mlir {
  namespace spn {
    namespace low {

      class Partition {

      public:

        Partition(unsigned ID, unsigned maximumSize) : id{ID}, numNodes{0}, maxSize{maximumSize} {
          // Allow up to 1% or at least one node in slack.
          unsigned slack = std::max(1u, static_cast<unsigned>(static_cast<double>(maxSize) * 0.01));
          sizeBoundary = maxSize + slack;
        };

        void addNode(Operation* node);

        void removeNode(Operation* node);

        bool contains(Operation* node);

        SmallPtrSetImpl<Operation*>::iterator begin();

        SmallPtrSetImpl<Operation*>::iterator end();

        llvm::ArrayRef<Operation*> hasExternalInputs();

        llvm::ArrayRef<Operation*> hasExternalOutputs();

        unsigned ID() const {
          return id;
        }

        unsigned size() const {
          return numNodes;
        }

        bool canAccept() const {
          return numNodes < sizeBoundary;
        }

        void dump() const;

      private:

        llvm::SmallPtrSet<Operation*, 32> nodes;

        bool dirty = false;

        llvm::SmallVector<Operation*> extIn;

        llvm::SmallVector<Operation*> exOut;

        void computeExternalConnections();

        unsigned id;

        unsigned numNodes;

        unsigned maxSize;

        unsigned sizeBoundary;

      };

      class GraphPartitioner {

      public:

        explicit GraphPartitioner(unsigned numberOfPartitions, HeuristicFactory heuristic = nullptr);

        Partitioning partitionGraph(llvm::ArrayRef<Operation*> nodes,
                                    llvm::ArrayRef<Operation*> inNodes,
                                    llvm::ArrayRef<Value> externalInputs);

      private:

        Partitioning initialPartitioning(llvm::ArrayRef<Operation*> nodes,
                                         llvm::ArrayRef<Operation*> inNodes,
                                         llvm::ArrayRef<Value> externalInputs) const;

        void refinePartitioning(llvm::ArrayRef<Operation*> allNodes, llvm::ArrayRef<Value> externalInputs,
                                Partitioning* allPartitions);

        bool hasInDegreeZero(Operation* node, llvm::SmallPtrSetImpl<Operation*>& partitioned,
                             llvm::SmallPtrSetImpl<Value>& externalInputs) const;

        unsigned numPartitions;

        HeuristicFactory factory;

      };

    }
  }
}

#endif //SPNC_MLIR_LIB_DIALECT_LOSPN_PARTITIONING_GRAPHPARTITIONER_H
