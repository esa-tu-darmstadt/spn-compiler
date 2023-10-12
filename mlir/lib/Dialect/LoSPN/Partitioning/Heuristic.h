//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_LIB_DIALECT_LOSPN_PARTITIONING_HEURISTIC_H
#define SPNC_MLIR_LIB_DIALECT_LOSPN_PARTITIONING_HEURISTIC_H

#include "GraphPartitioner.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include <memory>

namespace mlir {
namespace spn {
namespace low {
namespace partitioning {
// Forward declaration to avoid circular header dependency
class Partition;

class Heuristic {

public:
  Heuristic(llvm::ArrayRef<Node> allNodes, llvm::ArrayRef<Value> externalInputs, Partitioning *allPartitions);

  virtual ~Heuristic() = default;

  virtual void refinePartitioning() = 0;

protected:
  llvm::SmallVector<Node> nodes;

  llvm::SmallVector<Value> external;

  Partitioning *partitions;

  unsigned maxPartition = 0;

  llvm::DenseMap<Node, unsigned, llvm::DenseMapInfo<mlir::Operation*>> partitionMap;

  Partition *getPartitionForNode(Node node);

  unsigned getPartitionIDForNode(Node node);

  Partition *getPartitionByID(unsigned ID);

  void moveNode(Node node, Partition *from, Partition *to);

  bool isConstant(Node op) const;
};

class SimpleMoveHeuristic : public Heuristic {

public:
  using Heuristic::Heuristic;

  void refinePartitioning() override;

  static std::unique_ptr<SimpleMoveHeuristic>
  create(llvm::ArrayRef<Node> allNodes, llvm::ArrayRef<Value> externalInputs, Partitioning *allPartitions) {
    return std::make_unique<SimpleMoveHeuristic>(allNodes, externalInputs, allPartitions);
  }
};

} // namespace partitioning
} // namespace low
} // namespace spn
} // namespace mlir

#endif // SPNC_MLIR_LIB_DIALECT_LOSPN_PARTITIONING_HEURISTIC_H
