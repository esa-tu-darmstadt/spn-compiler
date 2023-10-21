//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#pragma once

#include "../../GraphPartitioner.h"

namespace mlir {
namespace spn {
namespace low {
namespace partitioning {
class TopologicalSortClustering : public GraphPartitioner::ClusteringAlgorithm {
  public:
    using GraphPartitioner::ClusteringAlgorithm::ClusteringAlgorithm;

    /// Perform the clustering
    void operator()(SPNGraph &graph) override;
};
} // namespace partitioning
} // namespace low
} // namespace spn
} // namespace mlir
