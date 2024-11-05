#pragma once

#include "../../GraphPartitioner.h"

namespace mlir {
namespace spn {
namespace low {
namespace partitioning {
/// Dominant sequence clustering algorithm.
class DominantSequenceClusteringScheduler
    : public GraphPartitioner::SchedulingAlgorithm {
public:
  // Inherit constructors
  using GraphPartitioner::SchedulingAlgorithm::SchedulingAlgorithm;

  /// Perform the scheduling
  Schedule operator()(SchedulingGraph &&graph) override;
};
} // namespace partitioning
} // namespace low
} // namespace spn
} // namespace mlir
