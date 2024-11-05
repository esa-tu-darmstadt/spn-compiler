//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================
#pragma once

#include "SPNGraph.h"
#include "Schedule.h"
#include "SchedulingGraph.h"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/properties.hpp>
#include <boost/pending/property.hpp>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include "llvm/Support/GraphWriter.h"

namespace mlir {
namespace spn {
namespace low {
namespace partitioning {

/// Represents a superstep in the BSP schedule. A superstep is a set of tasks
/// that can be executed in parallel.
class Superstep {
public:
  typedef std::shared_ptr<Superstep> Reference;

  explicit Superstep(superstep_index_t index) : index_(index) {}

  /// Returns the tasks and their assigned processors of this superstep.
  auto &tasks() const { return tasks_; }

  /// Returns the index of this superstep.
  superstep_index_t index() const { return index_; }

  // Returns the overall duration of this superstep. This includes the
  // communication and computation time and the synchronization time.
  int duration() const {
    return communicationAndComputationTime_ + synchronizationTime_;
  }

  /// Returns the communication and computation time of this superstep.
  int &communicationAndComputationTime() {
    return communicationAndComputationTime_;
  }

  /// Returns the synchronization time of this superstep.
  int &synchronizationTime() { return synchronizationTime_; }

  /// Returns or sets the processor of the given task.
  processor_t &operator[](cluster_index_t task) { return tasks_[task]; }

private:
  std::unordered_map<cluster_index_t, processor_t> tasks_;
  superstep_index_t index_;
  int communicationAndComputationTime_ = 0;
  int synchronizationTime_ = 0;
};

/// Represents a BSP schedule. The Bulk Synchronous Parallel (BSP) model is a
/// parallel programming model in which computation is divided into supersteps.
/// Supersteps consists of three phases: computation, communication, and
/// synchronization. All supersteps run synchronous on all processors, ie., they
/// begin at the same time and communication between processors is only possible
/// inbetween supersteps.
class BSPSchedule : public Schedule {
  /// Private constructor
  explicit BSPSchedule(Schedule &&schedule);

public:
  static BSPSchedule fromSchedule(Schedule &&schedule);

  void clusterGraph();

  /// Returns the supersteps of this schedule.
  auto &supersteps() const { return supersteps_; }

  /// Calculates the starting and ending times of the tasks in the schedule
  /// considering the BSP model.
  void calculateTimes() override;

private:
  std::vector<Superstep> supersteps_;
};

} // namespace partitioning
} // namespace low
} // namespace spn
} // namespace mlir