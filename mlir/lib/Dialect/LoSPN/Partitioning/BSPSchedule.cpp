#include "BSPSchedule.h"
#include "LoSPN/LoSPNAttributes.h"
#include "SchedulingGraph.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"
#include <numeric>
#include <unordered_map>

using namespace mlir::spn::low;
using namespace mlir::spn::low::partitioning;

#define DEBUG_TYPE "bsp-schedule"

BSPSchedule BSPSchedule::fromSchedule(Schedule &&nonBSPSchedule) {
  BSPSchedule schedule(std::move(nonBSPSchedule));

  LLVM_DEBUG(llvm::dbgs() << "Creating BSP schedule\n");

  // For each task, store the number of dependencies that are not yet scheduled
  std::unordered_map<cluster_index_t, int> outstandingDependencies;

  // For each processor, stores the number of tasks that are currently
  // scheduled. Used to determine the next task to schedule
  std::unordered_map<int, int> numScheduledTasks;

  // Initialize the outstanding dependencies
  for (auto vertex :
       boost::make_iterator_range(boost::vertices(schedule.graph_))) {
    cluster_index_t task =
        boost::get(SchedulingVertex_ClusterID(), schedule.graph_, vertex);
    outstandingDependencies[task] = boost::in_degree(vertex, schedule.graph_);
    LLVM_DEBUG(llvm::dbgs() << "Task " << vertex << " has "
                            << outstandingDependencies[vertex]
                            << " outstanding dependencies\n");
  }

  bool singleTaskInSuperstepPerProc = true;

  std::unordered_map<cluster_index_t, int>
      outstandDependenciesChangesAfterSuperstep;

  superstep_index_t superstepIndex = 0;
  bool maybeTaskAvailable = true;
  while (maybeTaskAvailable) {
    // Create a new superstep
    schedule.supersteps_.emplace_back(superstepIndex++);
    Superstep &superstep = schedule.supersteps_.back();

    maybeTaskAvailable = false;

    // Schedule the available tasks
    for (auto &[proc, tasks] : schedule.schedule_) {
      // Iterate over the tasks that are yet to be scheduled for this processor
      // in the order of the schedule
      for (auto currentTask = tasks.begin() + numScheduledTasks[proc];
           currentTask != tasks.end(); ++currentTask) {
        vertex_t vertex = *currentTask;
        cluster_index_t task =
            boost::get(SchedulingVertex_ClusterID(), schedule.graph_, vertex);
        if (outstandingDependencies[task] == 0) {
          // Schedule the task
          superstep[task] = proc;
          LLVM_DEBUG(llvm::dbgs()
                     << "Task " << task << " scheduled on processor " << proc
                     << " in superstep " << superstep.index() << "\n");
          numScheduledTasks[proc]++;

          // Decrease the number of outstanding dependencies for the successors.
          for (auto edge : boost::make_iterator_range(
                   boost::out_edges(vertex, schedule.graph_))) {
            vertex_t successor = boost::target(edge, schedule.graph_);
            cluster_index_t successorTask = boost::get(
                SchedulingVertex_ClusterID(), schedule.graph_, successor);
            // If the successor is on calculated on the same processor, the data
            // is directly available, within the same superstep. If the
            // successor is on a different processor, the data is available
            // after the superstep
            processor_t successorProc = boost::get(SchedulingVertex_ProcID(),
                                                   schedule.graph_, successor);
            if (successorProc == proc)
              --outstandingDependencies[successorTask];
            else
              ++outstandDependenciesChangesAfterSuperstep[successorTask];
            LLVM_DEBUG(llvm::dbgs() << "Task " << successorTask << " has now "
                                    << outstandingDependencies[successor]
                                    << " outstanding dependencies\n");
            // A new task may be available for scheduling
            maybeTaskAvailable = true;
          }
          if (singleTaskInSuperstepPerProc)
            break;
        } else {
          // Stop after the first task that cannot be scheduled is found. This
          // is required to keep the order of the tasks the same as in the
          // original schedule
          break;
        }
      }
    }

    // Update the outstanding dependencies now that the superstep is finished
    for (auto &[task, change] : outstandDependenciesChangesAfterSuperstep) {
      outstandingDependencies[task] -= change;
      change = 0;
      llvm::outs() << "Change is now "
                   << outstandDependenciesChangesAfterSuperstep[task] << "\n";
    }
  }

  // Now that all tasks are scheduled, we can update the starting and ending
  // times of each task
  schedule.calculateTimes();

  return schedule;
}

void BSPSchedule::calculateTimes() {
  // We define a superstep as 1. Communicate, 2. Compute, 3. Synchronize
  // So communication considers the data thats required by the current superstep
  int currentTime = 0;
  const int synchronizationDelay = 0;
  for (auto superstep : supersteps_) {
    int superstepCommAndCompTime = 0;
    for (auto [task, proc] : superstep.tasks()) {
      auto vertex = vertexOfTask[task];

      int taskCompTime = boost::get(vertex_weight(), graph_, vertex);
      int taskPreCommunicationTime = std::accumulate(
          boost::in_edges(vertex, graph_).first,
          boost::in_edges(vertex, graph_).second, 0, [&](int sum, auto edge) {
            return sum + boost::get(edge_weight(), graph_, edge);
          });

      // The task starts after the communication of the data it requires
      startingTimes_[proc].push_back(currentTime + taskPreCommunicationTime);
      endingTimes_[proc].push_back(currentTime + taskPreCommunicationTime +
                                   taskCompTime);

      // A superstep takes as long as the longest task in it
      superstepCommAndCompTime = std::max(
          superstepCommAndCompTime, taskCompTime + taskPreCommunicationTime);
    }
    superstep.communicationAndComputationTime() = superstepCommAndCompTime;
    superstep.synchronizationTime() = synchronizationDelay;

    currentTime += superstepCommAndCompTime + synchronizationDelay;
  }
}

BSPSchedule::BSPSchedule(Schedule &&schedule) : Schedule(std::move(schedule)) {}

void BSPSchedule::clusterGraph() {
  // Cluster the graph into subgraphs (for visualization)
  std::unordered_map<vertex_t, superstep_index_t> superstepOfTask;

  for (auto superstep : supersteps_) {
    for (auto [task, proc] : superstep.tasks()) {
      superstepOfTask[task] = superstep.index();
    }
  }

  // Add a subgraph for each superstep
  std::vector<SchedulingGraph *> subgraphOfSuperstep;
  subgraphOfSuperstep.reserve(supersteps_.size());
  for (size_t i = 0; i < supersteps_.size(); ++i) {
    SchedulingGraph &superstep = graph_.create_subgraph();
    subgraphOfSuperstep.push_back(&superstep);
    boost::get_property(superstep, SchedulingGraph_Superstep()) =
        supersteps_[i].index();
  }
  // Add the vertices to the subgraphs
  for (auto vertex : boost::make_iterator_range(boost::vertices(graph_))) {
    auto task = boost::get(SchedulingVertex_ClusterID(), graph_, vertex);
    auto superStep = superstepOfTask[task];
    auto &superstep = *subgraphOfSuperstep[superStep];
    boost::add_vertex(vertex, superstep);
  }
}

BSPScheduleAttr BSPSchedule::toAttr(MLIRContext *context) const {

  std::vector<Attribute> supersteps;
  supersteps.reserve(supersteps_.size());
  for (auto &superstep : supersteps_) {
    std::vector<int> taskIds, processorIDs;
    taskIds.reserve(superstep.tasks().size());
    processorIDs.reserve(superstep.tasks().size());

    for (auto &[task, proc] : superstep.tasks()) {
      taskIds.push_back(task);
      processorIDs.push_back(proc);
    }
    ShapedType indicesType = RankedTensorType::get(
        {(int64_t)taskIds.size()}, IntegerType::get(context, 32));
    DenseIntElementsAttr taskIdsAttr =
        DenseIntElementsAttr::get(indicesType, taskIds);
    DenseIntElementsAttr processorIDsAttr =
        DenseIntElementsAttr::get(indicesType, processorIDs);
    TaskProcessorMappingAttr tasks =
        TaskProcessorMappingAttr::get(context, taskIdsAttr, processorIDsAttr);
    supersteps.push_back(ArrayAttr::get(context, tasks));
  }
  return BSPScheduleAttr::get(context, ArrayAttr::get(context, supersteps));
}