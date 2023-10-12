#pragma once

#include "../../GraphPartitioner.h"

namespace mlir {
namespace spn {
namespace low {
namespace partitioning {
class Heuristic;
using HeuristicFactory =
    std::function<std::unique_ptr<Heuristic>(llvm::ArrayRef<Node>, llvm::ArrayRef<Value>, Partitioning *)>;

class TopoSortPartitioner : public GraphPartitioner {

  Partitioning initialPartitioning() const;
  void refinePartitioning(Partitioning *allPartitions);

  bool hasInDegreeZero(Node node, std::set<Node> &partitioned) const;


  HeuristicFactory factory;

public:
  using GraphPartitioner::GraphPartitioner;
  
  virtual Partitioning partitionGraph() override;

  virtual ~TopoSortPartitioner() = default;
};
} // namespace partitioning
} // namespace low
} // namespace spn
} // namespace mlir
