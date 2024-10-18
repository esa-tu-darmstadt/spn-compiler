#include "Schedule.h"
#include <algorithm>
#include <fstream>
#include <limits>
#include <unordered_map>

#include "BSPSchedule.h"
#include "SPNGraph.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"

using namespace mlir::spn::low::partitioning;

template <> void Schedule<BSPGraph>::calculateTimes() {
  // Calculate starting times for each processor

  // Store the tasks that have already been scheduled
  std::set<vertex_t> scheduledTasks;

  // Store the schedule in a deque to allow for efficient removal of the first element
  std::unordered_map<int, std::deque<vertex_t>> outstandingTasks;

  // Maps from task to processor
  std::unordered_map<vertex_t, int> procOfTask;

  // Maps from task to index in the schedule of its processor
  std::unordered_map<vertex_t, int> indexOfTask;

  // Copy the schedule
  for (auto &pair : schedule_) {
    auto proc = pair.first;
    outstandingTasks[pair.first] = std::deque<vertex_t>(pair.second.begin(), pair.second.end());

    // Fill the map from task to processor
    for (size_t i = 0; i < pair.second.size(); ++i) {
      vertex_t task = pair.second[i];
      procOfTask[task] = proc;
      indexOfTask[task] = i;
    }
  }
  bool allScheduled = false;
  while (!allScheduled) {
    allScheduled = true;

    // Try to schedule a free task
    for (auto &pair : outstandingTasks) {
      auto &outstandingTasksOfProc = pair.second;
      auto currentProc = pair.first;
      if (outstandingTasksOfProc.empty())
        continue; // This processor is done
      else
        allScheduled = false;

      auto nextTaskOfThisProc = outstandingTasksOfProc.front();

      // A task is ready if all its dependencies have been scheduled
      auto dependencies = boost::in_edges(nextTaskOfThisProc, graph_);
      bool ready = std::all_of(dependencies.first, dependencies.second, [&](auto edge) {
        auto source = boost::source(edge, graph_);
        return scheduledTasks.find(source) != scheduledTasks.end();
      });

      if (!ready)
        continue;

      auto &currentProcStartTimes = startingTimes_[currentProc];
      auto &currentProcEndTimes = endingTimes_[currentProc];

      // Find the earliest start time based of the dependencies
      float earliestStartTime = currentProcEndTimes.empty() ? 0 : currentProcEndTimes.back();
      for (auto edge : boost::make_iterator_range(dependencies.first, dependencies.second)) {
        auto source = boost::source(edge, graph_);
        auto sourceProc = procOfTask[source];
        auto sourceIndex = indexOfTask[source];
        auto sourceEndTime = endingTimes_[sourceProc][sourceIndex];

        // Add communication time if needed
        if (sourceProc != currentProc)
          sourceEndTime += boost::get(edge_weight(), graph_, edge);

        earliestStartTime = std::max(earliestStartTime, sourceEndTime);
      }

      // Schedule the task

      float endTime = earliestStartTime + boost::get(vertex_weight(), graph_, nextTaskOfThisProc);
      currentProcStartTimes.push_back(earliestStartTime);
      currentProcEndTimes.push_back(endTime);

      scheduledTasks.insert(nextTaskOfThisProc);
      outstandingTasksOfProc.pop_front();

      std::cout << "Scheduled task " << get_label(graph_, nextTaskOfThisProc) << " on processor " << currentProc
                << " at time " << earliestStartTime << std::endl;
    }
  }

  for (auto &pair : schedule_) {
    auto processor = pair.first;
    auto &tasks = pair.second;

    // Calculate starting times
    float time = 0;
    for (auto &task : tasks) {
      startingTimes_[processor].push_back(time);
      time += boost::get(vertex_weight(), graph_, task);
    }
  }
}
template <typename GraphT> int Schedule<GraphT>::makeSpan() {
  return std::max_element(endingTimes_.begin(), endingTimes_.end(),
                          [](const auto &a, const auto &b) { return a.second.back() < b.second.back(); })
      ->second.back();
}

template <typename GraphT>
void Schedule<GraphT>::saveAsHTML(std::string filename, const TargetExecutionModel &targetModel) {
  // Calculate starting times if not done yet
  if (startingTimes_.empty())
    calculateTimes();

  // Dump the schedule as a HTML file
  std::ofstream html(filename);

  auto timeToPixel = [](int time) { return time * 30; };
  auto durationOfVertex = [&](const auto &vertex) { return boost::get(vertex_weight(), graph_, vertex); };

  html << R"(
<html>
<head>
<style>
  body {
    font-family: Arial, sans-serif;
  }
  .schedule {
    display: flex;
  }
  .processor {
    box-sizing: border-box;
    width: 200px;
    height: 100%;
    background-color: black;
    float: left;
    position: relative;
  }
  .task {
    border: 1px solid black;
    box-sizing: border-box;
    left: 0;
    right: 0;
    background-color: white;
    position: absolute;
    display: flex;
    align-items: center;
    justify-content: center;
  }
</style>
</head>
<body>
)";

  auto currentTime = std::time(0);
  html << "<h1>Schedule</h1>\n";
  html << "<p>\n";
  html << "Generated by the SPN compiler on " << std::ctime(&currentTime) << "<br>\n";
  html << "Target: " << targetModel.getTargetName() << "<br>\n";
  html << "Makespan: " << makeSpan() << "<br>\n";
  html << "Number of processors: " << schedule_.size() << "<br>\n";
  html << "</p>\n";

  int headerHeight = 50;
  int scheduleHeight = timeToPixel(makeSpan()) + headerHeight;
  html << "<div class=\"schedule\" style=\"height: " << scheduleHeight << "px\">\n";

  for (const auto &pair : schedule_) {
    auto processor = pair.first;
    auto &tasks = pair.second;
    auto &startingTimes = startingTimes_[processor];

    html << "<div class=\"processor\">\n";
    html << "<div class=\"task\" style=\"height: " << headerHeight << "px; top: 0px; background-color: #C0C0C0;\">";
    html << "Processor " << processor;
    html << "</div>";
    for (size_t i = 0; i < tasks.size(); ++i) {
      auto vertex = tasks[i];
      auto start = startingTimes[i];
      auto top = timeToPixel(start) + headerHeight;
      auto height = timeToPixel(durationOfVertex(vertex));
      html << "<div class=\"task\" style=\"height: " << height << "px; top: " << top << "px\">";
      html << get_label(graph_, vertex);
      html << "</div>\n";
    }

    html << "</div>\n"; // Close processor div
  }

  html << "</div>\n"; // Close schedule div
  html << "</body>\n</html>";
  html.close();
}

template <class GraphT> void Schedule<GraphT>::viewSchedule(const TargetExecutionModel &targetModel) {
  int FD;
  SmallString<128> Filename;
  std::error_code EC = llvm::sys::fs::createTemporaryFile("schedule", "html", FD, Filename);
  if (EC) {
    llvm::errs() << "Error: " << EC.message() << "\n";
    return;
  }

  saveAsHTML(std::string(Filename.str()), targetModel);

  // Display the graph.
  EC = llvm::sys::fs::copy_file(Filename, "/workspaces/spn/schedule.html");
  // llvm::sys::ExecuteNoWait("firefox", { Filename }, {});
}

// Explicit template instantiations
template class mlir::spn::low::partitioning::Schedule<BSPGraph>;