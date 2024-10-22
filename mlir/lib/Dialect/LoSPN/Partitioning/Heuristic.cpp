// //==============================================================================
// // This file is part of the SPNC project under the Apache License v2.0 by the
// // Embedded Systems and Applications Group, TU Darmstadt.
// // For the full copyright and license information, please view the LICENSE
// // file that was distributed with this source code.
// // SPDX-License-Identifier: Apache-2.0
// //==============================================================================

// #include "Heuristic.h"
// #include "GraphPartitioner.h"
// #include "LoSPN/LoSPNOps.h"
// #include "mlir/IR/OpDefinition.h"
// #include "llvm/ADT/SmallPtrSet.h"

// using namespace mlir;
// using namespace mlir::spn::low;
// using namespace mlir::spn::low::partitioning;

// Heuristic::Heuristic(llvm::ArrayRef<Node> allNodes,
//                      llvm::ArrayRef<Value> externalInputs,
//                      Partitioning* allPartitions) : nodes(allNodes.begin(),
//                      allNodes.end()),
//                                                     external(externalInputs.begin(),
//                                                     externalInputs.end()),
//                                                     partitions(allPartitions)
//                                                     {
//   for (auto& p : *partitions) {
//     auto id = p->ID();
//     maxPartition = std::max(maxPartition, id);
//     for (auto& n : *p) {
//       partitionMap[n] = id;
//     }
//   }
// }

// unsigned int Heuristic::getPartitionIDForNode(Node node) {
//   assert(partitionMap.count(node) && "Node must be contained in map");
//   return partitionMap[node];
// }

// Partition* Heuristic::getPartitionForNode(Node node) {
//   auto partitionID = getPartitionIDForNode(node);
//   return partitions->at(partitionID).get();
// }

// Partition* Heuristic::getPartitionByID(unsigned int ID) {
//   assert(ID < partitions->size());
//   return partitions->at(ID).get();
// }

// void Heuristic::moveNode(Node node, Partition* from, Partition* to) {
//   from->removeNode(node);
//   to->addNode(node);
//   partitionMap[node] = to->ID();
// }

// bool Heuristic::isConstant(Node op) const {
//   return op->hasTrait<OpTrait::ConstantLike>();
// }

// void SimpleMoveHeuristic::refinePartitioning() {
//   llvm::SmallPtrSet<Value, 10> externalIn(external.begin(), external.end());

//   for (auto& node : nodes) {

//     // Do not move constant operations, they can be internalized by
//     duplication. if (node->hasTrait<OpTrait::ConstantLike>()) {
//       continue;
//     }

//     // Each node can either be moved from partition i 'upward' to partition
//     i+1
//     // or 'downward' to partition i-1.
//     auto i = getPartitionIDForNode(node);

//     // Upward direction
//     llvm::Optional<int> upwardCost = llvm::None;
//     if (i != maxPartition) {
//       // Calculate the positive gain: All edges to users of 'node' that are
//       in partition i+1
//       // will become internal edges.
//       int gain = 0;
//       bool legal = true;
//       for (auto* U : node->getUsers()) {
//         // SPNYield is not part of the partitioning, simply ignore it.
//         if (isa<SPNYield>(U)) {
//           continue;
//         }
//         auto partitionID = getPartitionIDForNode(U);
//         if (partitionID == i + 1) {
//           // Although there might be multiple users in the upward partition,
//           the overall gain for all
//           // of them combined is only 1, as the result of the node only needs
//           to be stored/loaded to/from
//           // memory once for all users in one partition.
//           gain = 1;
//         }
//         if (partitionID == i) {
//           // If we have an user in the same partition, it's not legal to move
//           the node upwards. legal = false; break;
//         }
//       }
//       // Calculate the negative gain. All edges from operands of 'node' that
//       are in partition i
//       // will become external edges.
//       for (auto operand : node->getOperands()) {
//         if (externalIn.contains(operand)) {
//           continue;
//         }
//         auto defOp = operand.getDefiningOp();
//         if (!isConstant(defOp) && getPartitionIDForNode(defOp) == i) {
//           --gain;
//         }
//       }
//       if (legal) {
//         upwardCost = gain;
//       }
//     }

//     // Downward direction
//     llvm::Optional<int> downwardCost = llvm::None;
//     if (i != 0) {
//       // Calculate the positive gain: All edges from operands of 'node' that
//       are in partition i-1
//       // will become internal edges.
//       int gain = 0;
//       bool legal = true;
//       for (auto operand : node->getOperands()) {
//         if (externalIn.contains(operand)) {
//           continue;
//         }
//         auto defOp = operand.getDefiningOp();
//         auto partitionID = getPartitionIDForNode(defOp);
//         if (!isConstant(defOp) && partitionID == i - 1) {
//           ++gain;
//         }
//         if (!isConstant(defOp) && partitionID == i) {
//           // If we have an operand in the same partition, it's illegal to
//           push the node downwards. legal = false; break;
//         }
//       }
//       // Calculate the negative gain: All edges to users of 'node' that are
//       in partition i
//       // will become external edges. As the result of the operation only
//       needs to be
//       // stored/loaded to/from memory once for all users, the gain is only
//       decremented by 1
//       // if there's any user.
//       auto usedInSame = llvm::any_of(node->getUsers(), [this, i](Node U) {
//         // SPNYield is not part of the partitioning, simply ignore it.
//         return !isa<SPNYield>(U.getOperation()) && getPartitionIDForNode(U)
//         == i;
//       });
//       if (usedInSame) {
//         --gain;
//       }

//       if (legal) {
//         downwardCost = gain;
//       }
//     }

//     int upwardGain = upwardCost.getValueOr(INT32_MIN);
//     int downwardGain = downwardCost.getValueOr(INT32_MIN);
//     auto partition = getPartitionForNode(node);
//     if (upwardGain == downwardGain && upwardGain >= 0) {
//       // We have a tie, with positive gain in both cases.
//       // Try to resolve the tie based on size of the partitions.
//       auto* upwardPart = getPartitionByID(i + 1);
//       auto* downwardPart = getPartitionByID(i - 1);
//       auto upwardDifference = static_cast<int>(partition->size()) -
//       static_cast<int>(upwardPart->size()); auto downwardDifference =
//       static_cast<int>(partition->size()) -
//       static_cast<int>(downwardPart->size()); if (upwardDifference >
//       downwardDifference && (upwardDifference > 0 || upwardGain > 0)
//           && upwardPart->canAccept()) {
//         moveNode(node, partition, upwardPart);
//       } else if ((downwardDifference > 0 || downwardGain > 0) &&
//       downwardPart->canAccept()) {
//         moveNode(node, partition, downwardPart);
//       }
//     }
//     if (upwardGain > downwardGain && upwardGain >= 0) {
//       auto* upwardPart = getPartitionByID(i + 1);
//       auto upwardDifference = static_cast<int>(partition->size()) -
//       static_cast<int>(upwardPart->size()); if ((upwardGain > 0 ||
//       upwardDifference > 0) && upwardPart->canAccept()) {
//         // Operations are moved in two cases:
//         // (1) We have a positive gain (i.e. less edges crossing partitions)
//         // (2) We can balance the partitions (i.e., partition i is larger
//         than partition i+1). moveNode(node, partition, upwardPart);
//       }
//     }
//     if (downwardGain > upwardGain && downwardGain >= 0) {
//       auto* downwardPart = getPartitionByID(i - 1);
//       auto downwardDifference = static_cast<int>(partition->size()) -
//       static_cast<int>(downwardPart->size()); if ((downwardGain > 0 ||
//       downwardDifference > 0) && downwardPart->canAccept()) {
//         // Operations are moved in two cases:
//         // (1) We have a positive gain (i.e. less edges crossing partitions)
//         // (2) We can balance the partitions (i.e., partition i is larger
//         than partition i+1). moveNode(node, partition, downwardPart);
//       }
//     }
//   }

// }
