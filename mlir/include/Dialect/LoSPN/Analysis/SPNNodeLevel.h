//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_LOSPN_ANALYSIS_SPNNODELEVEL_H
#define SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_LOSPN_ANALYSIS_SPNNODELEVEL_H

#include "LoSPN/LoSPNOps.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir {
namespace spn {

///
/// Analysis to compute the distance of nodes to the root node in an SPN graph.
class SPNNodeLevel {

public:
  /// Constructor, initialize analysis.
  /// \param root Root node of a (sub-)graph or query operation.
  /// \param rootLevel Level to assign to root node, defaults to 0.
  explicit SPNNodeLevel(Operation *root, int rootLevel = 0);

  /// Get distance of operation to (sub-)graph root.
  /// \param op Operation.
  /// \return Depth, i.e. distance from the root node.
  int getOperationDepth(Operation *op) const;

  ///
  /// \return Maximum depth of a leaf node.
  int getMaxDepth() const;

  ///
  /// \return Minimum depth of a leaf node.
  int getMinDepth() const;

  ///
  /// \return Median depth of the leaf nodes.
  double getMedianDepth() const;

  ///
  /// \return Average depth of the leaf nodes.
  double getAverageDepth() const;

  /// Get the complete result computed for this (sub-)graph.
  /// \return Mapping from operation to depth.
  const DenseMap<Operation *, int> &getFullResult() const;

  /// Root node used for this analysis.
  /// \return Root node of the analyzed (sub-)graph.
  const Operation *getRootNode() const;

private:
  struct GraphLevelInfo {
    int level;
  };

  void analyzeGraph(Operation *graphRoot);

  void traverseSubgraph(Operation *subgraphRoot, GraphLevelInfo info);

  Operation *rootNode;

  int rootNodeLevel;

  int maxDepth = 0;
  int minDepth = std::numeric_limits<int>::max();

  double averageDepth = 0;
  double medianDepth = 0;

  DenseMap<Operation *, int> opLevels;

  std::set<int> leafLevels;

  llvm::IndexedMap<int> leafLevelHistogram;

  unsigned numLeafNodes = 0;
};

} // namespace spn
} // namespace mlir

#endif // SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_LOSPN_ANALYSIS_SPNNODELEVEL_H
