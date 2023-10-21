#pragma once

#include "../../GraphPartitioner.h"

namespace mlir {
namespace spn {
namespace low {
namespace partitioning {
class DominantSequenceClustering : public GraphPartitioner::ClusteringAlgorithm {
  public:
    using GraphPartitioner::ClusteringAlgorithm::ClusteringAlgorithm;

    /// Perform the clustering
    void operator()(SPNGraph &graph) override;
};
} // namespace partitioning
} // namespace low
} // namespace spn
} // namespace mlir
