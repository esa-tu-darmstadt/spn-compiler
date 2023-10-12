#pragma once

#include "../../GraphPartitioner.h"
#include <unordered_map>

namespace mlir {
namespace spn {
namespace low {
namespace partitioning {

class DSCPartitioner : public GraphPartitioner {
public:
using GraphPartitioner::GraphPartitioner;


    Partitioning partitionGraph() override;

private:
    // A map to check if a node has been visited during the longest path search
    std::unordered_map<Node, bool> visited_;

    // A recursive function to find the longest path from a given node
    float findLongestPath(Node node, std::vector<Node> &path);

public:

};

} // namespace partitioning
} // namespace low
} // namespace spn
} // namespace mlir