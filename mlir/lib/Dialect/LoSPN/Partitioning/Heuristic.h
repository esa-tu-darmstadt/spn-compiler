//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_LIB_DIALECT_LOSPN_PARTITIONING_HEURISTIC_H
#define SPNC_MLIR_LIB_DIALECT_LOSPN_PARTITIONING_HEURISTIC_H

#include <memory>
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
  namespace spn {
    namespace low {

      // Forward declaration to avoid circular header dependency
      class Partition;

      using PartitionRef = std::unique_ptr<Partition>;

      using Partitioning = std::vector<PartitionRef>;

      class Heuristic {

      public:

        Heuristic(llvm::ArrayRef<Operation*> allNodes, llvm::ArrayRef<Value> externalInputs,
                  Partitioning* allPartitions);

        virtual ~Heuristic() = default;

        virtual void refinePartitioning() = 0;

      protected:

        llvm::SmallVector<Operation*> nodes;

        llvm::SmallVector<Value> external;

        Partitioning* partitions;

        unsigned maxPartition = 0;

        llvm::DenseMap<Operation*, unsigned> partitionMap;

        Partition* getPartitionForNode(Operation* node);

        unsigned getPartitionIDForNode(Operation* node);

        Partition* getPartitionByID(unsigned ID);

        void moveNode(Operation* node, Partition* from, Partition* to);

      };

      using HeuristicFactory =
      std::function<std::unique_ptr<Heuristic>(llvm::ArrayRef<Operation*>, llvm::ArrayRef<Value>,
                                               Partitioning*)>;

      class SimpleMoveHeuristic : public Heuristic {

      public:

        using Heuristic::Heuristic;

        void refinePartitioning() override;

        static std::unique_ptr<SimpleMoveHeuristic> create(llvm::ArrayRef<Operation*> allNodes,
                                                           llvm::ArrayRef<Value> externalInputs,
                                                           Partitioning* allPartitions) {
          return std::make_unique<SimpleMoveHeuristic>(allNodes, externalInputs, allPartitions);
        }

      };

    }
  }
}

#endif //SPNC_MLIR_LIB_DIALECT_LOSPN_PARTITIONING_HEURISTIC_H
