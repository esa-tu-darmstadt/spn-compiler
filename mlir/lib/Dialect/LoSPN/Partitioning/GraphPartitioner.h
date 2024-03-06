//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_LIB_DIALECT_LOSPN_PARTITIONING_GRAPHPARTITIONER_H
#define SPNC_MLIR_LIB_DIALECT_LOSPN_PARTITIONING_GRAPHPARTITIONER_H

#include "Heuristic.h"
#include "LoSPN/LoSPNOps.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {
namespace spn {
namespace low {

class Partition {

public:
  Partition(unsigned ID, unsigned maximumSize)
      : id{ID}, numNodes{0}, sizeBoundary{maximumSize} {};

  void addNode(Operation *node);

  void removeNode(Operation *node);

  bool contains(Operation *node);

  SmallPtrSetImpl<Operation *>::iterator begin();

  SmallPtrSetImpl<Operation *>::iterator end();

  llvm::ArrayRef<Operation *> hasExternalInputs();

  llvm::ArrayRef<Operation *> hasExternalOutputs();

  unsigned ID() const { return id; }

  unsigned size() const { return numNodes; }

  bool canAccept() const { return numNodes < sizeBoundary; }

  void invalidateExternal() { dirty = true; }

  void dump() const;

private:
  llvm::SmallPtrSet<Operation *, 32> nodes;

  bool dirty = false;

  llvm::SmallVector<Operation *> extIn;

  llvm::SmallVector<Operation *> exOut;

  void computeExternalConnections();

  unsigned id;

  unsigned numNodes;

  unsigned sizeBoundary;
};

class GraphPartitioner {

public:
  explicit GraphPartitioner(int maxTaskSize,
                            HeuristicFactory heuristic = nullptr);

  Partitioning partitionGraph(llvm::ArrayRef<Operation *> nodes,
                              llvm::SmallPtrSetImpl<Operation *> &inNodes,
                              llvm::ArrayRef<Value> externalInputs);

  unsigned getMaximumPartitionSize() const;

private:
  Partitioning initialPartitioning(llvm::ArrayRef<Operation *> nodes,
                                   llvm::SmallPtrSetImpl<Operation *> &inNodes,
                                   llvm::ArrayRef<Value> externalInputs) const;

  void refinePartitioning(llvm::ArrayRef<Operation *> allNodes,
                          llvm::ArrayRef<Value> externalInputs,
                          Partitioning *allPartitions);

  bool hasInDegreeZero(Operation *node,
                       llvm::SmallPtrSetImpl<Operation *> &partitioned,
                       llvm::SmallPtrSetImpl<Value> &externalInputs) const;

  int maxPartitionSize;

  HeuristicFactory factory;
};

} // namespace low
} // namespace spn
} // namespace mlir

#endif // SPNC_MLIR_LIB_DIALECT_LOSPN_PARTITIONING_GRAPHPARTITIONER_H
