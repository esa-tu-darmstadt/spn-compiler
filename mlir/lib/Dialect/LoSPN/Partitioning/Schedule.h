#pragma once

#include "SchedulingGraph.h"
#include "TargetExecutionModel.h"
#include <boost/graph/adjacency_list.hpp>
#include <unordered_map>

namespace mlir {
namespace spn {
namespace low {
namespace partitioning {
class Schedule {
protected:
  using vertex_t =
      typename boost::graph_traits<SchedulingGraph>::vertex_descriptor;
  // Map from processor to list of vertices scheduled on that processor in order
  std::unordered_map<int, std::vector<int>> schedule_;

  //  Map from processor to list of starting times of each task on that
  //  processor
  std::unordered_map<int, std::vector<int>> startingTimes_;

  //  Map from processor to list of end times of each task on that processor
  std::unordered_map<int, std::vector<int>> endingTimes_;

  // Map from a cluster index to the vertex representing the task in the
  // scheduling graph
  std::unordered_map<cluster_index_t, vertex_t> vertexOfTask;

  SchedulingGraph graph_;

public:
  // Creates an empty schedule and populates the `vertexOfTask` map
  Schedule(SchedulingGraph &&graph);
  Schedule(Schedule &&) = default;

  virtual ~Schedule() = default;

  // Returns the schedule for a processor
  auto &operator[](int processor) { return schedule_[processor]; }

  auto &schedule() { return schedule_; }
  auto &startingTimes() { return startingTimes_; }
  int makeSpan();

  SchedulingGraph &graph() { return graph_; }

  /// Calculates the starting and ending times of each task on each processor
  virtual void calculateTimes();

  /// Updates underlying graph with the schedule information (processor,
  /// starting time, etc)
  void updateGraph();

  void viewSchedule(const TargetExecutionModel &targetModel, std::string title,
                    std::string filename);
  void saveAsHTML(std::string filename, const TargetExecutionModel &targetModel,
                  std::string title);
};
} // namespace partitioning
} // namespace low
} // namespace spn
} // namespace mlir