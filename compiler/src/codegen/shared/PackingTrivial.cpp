#include "PackingTrivial.h"
#include "codegen/shared/BFSOrderProducer.h"
#include "codegen/shared/VectorizationTraversal.h"
#include <map>
#include <iostream>

#define MIN_LENGTH 2

std::vector<std::vector<NodeReference>>
PackingTrivial::getLongestChain(std::vector<NodeReference> vectorRoots,
                             std::unordered_set<std::string> pruned, size_t width) {
  // lexicographically sorted, thus paths with longest shared prefixes are next
  // to each other
  std::multimap<std::string, std::pair<size_t, size_t>> paths;
  std::vector<std::vector<std::vector<NodeReference>>> raw_paths;
  auto trav = VectorizationTraversal(pruned);
  for (unsigned i = 0; i < vectorRoots.size(); i++) {
    auto res = trav.collectPaths(vectorRoots[i]);
    raw_paths.push_back({});
    for (int j = 0; j < res.size(); j++) {
      paths.insert({res[j].first, {i, j}});
      raw_paths[i].push_back(res[j].second);
    }
  }

  int maxPrefixLength = -1;
  std::vector<decltype(paths.rbegin())> maxPaths;

  /* Use reverse iterator so that for two paths starting at the same root, where
     one path's operation order is a prefix of the path's operation order, the
     longer path is considered
  */
  for (auto rIt = paths.rbegin(); std::distance(rIt, paths.rend()) >= width; rIt++) {
    std::unordered_map<size_t, decltype(rIt)> independentPaths;
    independentPaths.insert({rIt->second.first, rIt});
    auto candidate = std::next(rIt);
    // Find width independent paths that are lexicographically smaller than
    // rIt's path but lexicographically as close as possible to rIt's path
    while (independentPaths.size() < width && candidate != paths.rend()) {
      if (independentPaths.find(candidate->second.first) ==
          independentPaths.end()) {
	// found the next smaller path to rIt's for a root, that has not been selected yet
	independentPaths.insert({candidate->second.first, candidate});
      }
      candidate++;
    }

    if (independentPaths.size() < width) {
      continue;
    }

    // Now check, how long the common prefix of the paths in independentPaths is

    auto it = independentPaths.begin();
    std::string prefix = it->second->first;

    for (it = std::next(it); it != independentPaths.end(); it++) {
      std::string newPrefix;
      int preLength = prefix.length(), pathLength = it->second->first.length();
      for (int i = 0; i < preLength && i < pathLength; i++) {
        if (prefix[i] != it->second->first[i])
          break;
        newPrefix.push_back(prefix[i]);
      }
      prefix = newPrefix;
    }
    int preLen = prefix.length();
    if (preLen > maxPrefixLength) {
      // Found a longer isomorphic chain
      maxPrefixLength = prefix.length();
      std::vector<decltype(paths.rbegin())> newPaths;
      for (auto& e : independentPaths) {
	newPaths.push_back(e.second);
      }
      maxPaths = newPaths;
    }
    
  }

  if (maxPrefixLength < MIN_LENGTH) {
    return {};
  }

  // TODO: Make the inner an array
  std::vector<std::vector<NodeReference>> nodeGroupSequence(maxPrefixLength);
  
  for (int i = 0; i < width; i++) {
    auto &nodeVec =
        raw_paths[maxPaths[i]->second.first][maxPaths[i]->second.second];
    for (int j = 0; j < maxPrefixLength; j++) {
      nodeGroupSequence[j].push_back(nodeVec[j]);
    }
  }

  return nodeGroupSequence;
}

struct rootSetInfo {
  std::vector<NodeReference> rootNodes;
  std::unordered_set<std::string> prunedNodes;
  // Indices into _sequences_, pointing to selected pack, which these roots are
  // a direct input into
  std::pair<int, int> directInput;
};

vectorizationResultInfo PackingTrivial::getVectorization(IRGraph& graph, size_t width) {
  // Perform BFS to find starting tree level
  std::vector<NodeReference> vectorRoots;
  BFSOrderProducer visitor;
  graph.rootNode->accept(visitor, {});
  while (!visitor.q.empty()) {
    if (visitor.currentLevel < visitor.q.front().first) {
      if (vectorRoots.size() >= width) {
        break;
      } else {
        visitor.currentLevel++;
	vectorRoots.clear();
      }
    }
    visitor.q.front().second->accept(visitor, {});
    vectorRoots.push_back(visitor.q.front().second);
    visitor.q.pop();
  }

  if (vectorRoots.size() < width)
    return {};
  
  std::queue<rootSetInfo> rootSetQueue;
  
  rootSetQueue.push({vectorRoots, {}, {-1, 0}});
  // sequences[second][0] is direct input for sequences[first.first][first.second]
  std::unordered_map<size_t, std::unordered_map<size_t, std::vector<size_t>>> directInputs;

  std::vector<std::vector<std::vector<NodeReference>>> sequences;
  std::unordered_set<size_t> dependentChains;
  while (!rootSetQueue.empty()) {
    auto roots = rootSetQueue.front().rootNodes;
    std::unordered_set<std::string> pruned = rootSetQueue.front().prunedNodes;
    auto nodeGroupSequence = getLongestChain(roots, pruned, width);

    std::pair<int, int> directIdx = {-1,0};
    while (nodeGroupSequence.size() > 0) {
      sequences.push_back(nodeGroupSequence);
      if (directIdx.first == -1)
	directIdx = rootSetQueue.front().directInput;
      if (directIdx.first != -1) {
	directInputs[directIdx.first][directIdx.second].push_back(sequences.size()-1);
	dependentChains.insert(sequences.size()-1);
      }
      
      if (roots.size() > width) {
	// Prevent other groupings than the one in nodeGroupSequence[0] by splitting the roots set
        roots.erase(std::remove_if(roots.begin(), roots.end(),
                                   [&](auto &nr) {
                                     for (auto &n : nodeGroupSequence[0]) {
                                       if (nr->id() == n->id()) {
                                         return true;
                                       }
                                     }
                                     return false;
                                   }),
                    roots.end());

        std::unordered_set<std::string> prunesOfSplit;
	for (auto &n : nodeGroupSequence[1]) {
          prunesOfSplit.insert(n->id());
        }
        rootSetQueue.push({nodeGroupSequence[0], prunesOfSplit,
                           rootSetQueue.front().directInput});
      } else {
        for (auto &n : nodeGroupSequence[1]) {
          pruned.insert(n->id());
        }
	if (directIdx.first == -1) {
          // we just fixed the very first sequence, so all other sequences
          // starting from this root will be direct vec inputs to the first
          // vector of the just found sequence
          directIdx = {sequences.size()-1, 0};
	}
      }

      // add new root sets along the chain
      for (int i = 1; i + 1 < nodeGroupSequence.size(); i++) {
        std::vector<NodeReference> newRoots;
        for (auto &n : nodeGroupSequence[i]) {
          newRoots.push_back(n);
        }
        std::unordered_set<std::string> newPrunes;
        for (auto &n : nodeGroupSequence[i+1]) {
          newPrunes.insert(n->id());
        }
        rootSetQueue.push({newRoots, newPrunes, {sequences.size()-1, i}});
      }

      std::cout << "new chain " << std::endl;
      for (auto &e : nodeGroupSequence) {
        std::cout << "new ins " << std::endl;
        for (auto &n : e)
          std::cout << "id " << n->id() << std::endl;
      }

      nodeGroupSequence = getLongestChain(roots, pruned, width);
    }
    rootSetQueue.pop();
  }
  vectorizationResultInfo res;
  std::unordered_map<size_t, std::unordered_map<size_t, size_t>> seqIdxToVectorsIdx;
  for (int i = 0; i < sequences.size(); i++) {
    auto & seq = sequences[i];
    for (int j = 0 ; j < seq.size(); j++) {
      if (j == 0 && dependentChains.find(i) != dependentChains.end()) {
        // The first element of this sequence was already created when handling
        // the sequence that this sequence originated from
	continue;
      }
      auto& instr = seq[j];
      res.vectors.push_back(instr);
      seqIdxToVectorsIdx[i][j] = res.vectors.size() - 1;
      for (auto& lane : instr) {
	res.partOf.insert({lane->id(), res.vectors.size()-1});
      }

      if ((j > 0 && dependentChains.find(i) == dependentChains.end()) || j > 1) {
        // In a sequence, each successor is a direct vector input to is
        // predecessor, since the sequences are built from the root to the leafs
        res.directVecInputs[res.vectors.size() - 2].insert(res.vectors.size() -
                                                           1);
      }
    }
  }

  for (auto &seq : directInputs) {
    for (auto &vec : seq.second) {
      size_t vectorIdx = seqIdxToVectorsIdx[seq.first][vec.first];
      for (auto vecInput : vec.second) {
        size_t inputVectorIdx = seqIdxToVectorsIdx[vecInput][1];
        res.directVecInputs[vectorIdx].insert(inputVectorIdx);
      }
    }
  }
  return res;
}
