//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_LOSPN_ANALYSIS_SPNGRAPHSTATISTICS_H
#define SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_LOSPN_ANALYSIS_SPNGRAPHSTATISTICS_H

#include "llvm/ADT/StringMap.h"
#include "LoSPN/LoSPNOps.h"

namespace mlir {
  namespace spn {

    ///
    /// Analysis to compute static graph properties of an SPN graph.
    class SPNGraphStatistics {

    public:

      /// Constructor, initialize analysis.
      /// \param root Root node of a (sub-)graph or query operation.
      explicit SPNGraphStatistics(Operation* root);

      ///
      /// \return Overall number of nodes in the (sub-)graph.
      unsigned getNodeCount() const;

      /// Get count for one kind of nodes.
      /// \tparam Kind Operation-class
      /// \return Number of nodes of this kind in the (sub-)graph.
      template<typename Kind>
      unsigned getKindNodeCount() const {
        if (nodeKindCount.count(Kind::getOperationName())) {
          return nodeKindCount.lookup(Kind::getOperationName());
        }
        return 0;
      }

      ///
      /// \return Number of inner (non-leaf) nodes in the (sub-)graph.
      unsigned getInnerNodeCount() const;

      ///
      /// \return Number of leaf nodes in the (sub-)graph.
      unsigned getLeafNodeCount() const;

      /// Full statistics for this (sub-)graph.
      /// \return Mapping from operation name to count.
      const llvm::StringMap<unsigned int>& getFullResult() const;

      ///
      /// \return Root node of the analyzed (sub-)graph.
      const Operation* getRootNode() const;

    private:

      void analyzeGraph(Operation* root);

      Operation* rootNode;

      int innerNodeCount = 0;

      int leafNodeCount = 0;

      llvm::StringMap<unsigned int> nodeKindCount;

    };

  }
}

#endif //SPNC_MLIR_DIALECTS_INCLUDE_DIALECT_LOSPN_ANALYSIS_SPNGRAPHSTATISTICS_H
