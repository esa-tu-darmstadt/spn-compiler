#pragma once

#include "../../GraphPartitioner.h"

namespace mlir {
namespace spn {
namespace low {
namespace partitioning {
/// Dominant sequence clustering algorithm.
class DominantSequenceClusteringPartitioner
    : public GraphPartitioner::ClusteringAlgorithm {
public:
  // Inherit constructors
  using GraphPartitioner::ClusteringAlgorithm::ClusteringAlgorithm;

  /// Perform the clustering
  void operator()(SPNGraph &graph) override;
};

template <typename GraphT>
class DominantSequenceClusteringScheduler
    : public GraphPartitioner::SchedulingAlgorithm<GraphT> {
public:
  // Inherit constructors
  using GraphPartitioner::SchedulingAlgorithm<GraphT>::SchedulingAlgorithm;

  /// Perform the scheduling
  Schedule<GraphT> operator()(GraphT &graph) override;
};
} // namespace partitioning
} // namespace low
} // namespace spn
} // namespace mlir
