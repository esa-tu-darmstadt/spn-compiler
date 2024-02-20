#pragma once

#include "TargetExecutionModel.h"
#include <boost/graph/adjacency_list.hpp>
#include <unordered_map>

namespace mlir {
namespace spn {
namespace low {
namespace partitioning {
template <class GraphT> class Schedule {
  typedef typename boost::graph_traits<GraphT>::vertex_descriptor vertex_t;

  // Map from processor to list of vertices scheduled on that processor in order
  std::unordered_map<int, std::vector<vertex_t>> schedule_;

  //  Map from processor to list of starting times of each task on that processor
  std::unordered_map<int, std::vector<float>> startingTimes_;

  //  Map from processor to list of end times of each task on that processor
  std::unordered_map<int, std::vector<float>> endingTimes_;

  GraphT &graph_;

public:
  Schedule(GraphT &graph) : graph_(graph) {}
  auto &operator[](int processor) { return schedule_[processor]; }

  auto &schedule() { return schedule_; }
  auto &startingTimes() { return startingTimes_; }
  int makeSpan();

  void calculateTimes();


  void viewSchedule(const spnc::TargetExecutionModel &targetModel);
  void saveAsHTML(std::string filename, const spnc::TargetExecutionModel &targetModel);
};
} // namespace partitioning
} // namespace low
} // namespace spn
} // namespace mlir