#include "PackingHeuristic.h"
#include <iostream>
#include <queue>

class SerialCostCalc : public BaseVisitor {
public:
  SerialCostCalc(CostInfo *ci_) : ci(ci_) {}
  size_t getCost(const NodeReference &rootNode, std::unordered_map<std::string, size_t>* vectorized_) {
    vectorized = vectorized_;
    auto res = getCost(rootNode);
    vectorized = nullptr;
    return res;
  }

  
  size_t
  getCost(const NodeReference &rootNode) {
    curCost = 0;
    rootNode->accept(*this, {});
    return curCost;
  }

  void visitHistogram(Histogram &n, arg_t arg)  {
  }

  void visitGauss(Gauss &n, arg_t arg) {
    curCost += ci->gaussArithCost;
  }

  void visitProduct(Product &n, arg_t arg) {
    for (auto &c : n.multiplicands()) {
      if (vectorized && vectorized->find(c->id()) != vectorized->end()) {
	// extract and arith
	curCost += ci->getExtractCost((*vectorized)[c->id()]) + ci->scalarArithCost;
      } else {
        curCost += ci->scalarArithCost;
        c->accept(*this, {});
      }
    }
  }

  void visitSum(Sum &n, arg_t arg)  {
    for (auto &c : n.addends()) {
      if (vectorized && vectorized->find(c->id()) != vectorized->end()) {
	// extract and arith
	curCost += ci->getExtractCost((*vectorized)[c->id()]) + ci->scalarArithCost;
      } else {
        curCost += ci->scalarArithCost;
        c->accept(*this, {});
      }
    }
  }

  void visitWeightedSum(WeightedSum &n, arg_t arg)  {
    for (auto &c : n.addends()) {
      if (vectorized && vectorized->find(c.addend->id()) != vectorized->end()) {
	// extract and arith
	curCost += ci->getExtractCost((*vectorized)[c.addend->id()]) + ci->scalarArithCost;
      } else {
        curCost += ci->scalarArithCost;
        c.addend->accept(*this, {});
      }
    }
  }

private:
  size_t curCost;
  std::unordered_map<std::string, size_t>* vectorized = nullptr;
  CostInfo* ci;
};

// TODO: Branch and cut when evaluating different options for packs for
// (chain, pos) also save best config for already handled chain, so that
// calc'ing chainSet cost is after some time just adding up chain cost
// TODO candidates is actually always a range of integers, we don't need a
// vector here
setSolverRes PackingHeuristic::returnBestSet(
    std::vector<size_t> candidates, std::vector<SIMDChainSet> &simdChainSets,
    InitialChainBuilder &icb,
    std::unordered_map<size_t, std::unordered_map<size_t, std::vector<size_t>>>
        &chainPosToPackVecMap,
    std::vector<size_t> source, std::set<size_t> pruned) {

  SerialCostCalc scc(ci.get());
  
  size_t bestCandidateCost = 0;
  std::vector<vectorizationResultInfo> bestSubTrees;
  if (source.size() < 2) {
    bestCandidateCost = scc.getCost(icb.nodes[source[0]]);
  } else {
    for (auto &lane : source) {
      std::vector<size_t> serNodes;

      auto &laneChildren = icb.parentChildrenMap[lane];
      std::set_difference(laneChildren.begin(), laneChildren.end(),
                          pruned.begin(), pruned.end(),
                          std::inserter(serNodes, serNodes.begin()));

      if (serNodes.size() > 0) {
        // cost of insertion
        bestCandidateCost += ci->getInsertCost(source.size());
        // cost of scalar arithmetic operations for producing lane
        bestCandidateCost += (serNodes.size() - 1)*ci->scalarArithCost;
        for (auto &child : serNodes) {
	  auto childRes = getVectorizationRec(child, icb);
          bestCandidateCost += childRes.second;
	  bestSubTrees.push_back(childRes.first);
        }
      }
    }
    if (pruned.size() != 0) {
      // cost of mul/add of pruned vector to source vector
      bestCandidateCost += ci->vecArithCost;
    }
  }
  
  std::vector<size_t> bestSelectedSets;

  for (auto& cand : candidates) {
    size_t candidateCost = 0;
    std::vector<vectorizationResultInfo> candidateSubtrees;
    std::vector<size_t> selectedSetsForCand;
    for (auto& chain : simdChainSets[cand].SIMDChains) {
      if (source.size() < 2) {
        // cost of arith to source for vector produced by chain
        // if this is the initial call to returnBestSet (i.e. source.size() <
        // 2), the cost of extracting the result of chain will be accounted for
        // in serialCostCalc
        candidateCost += ci->vecArithCost;
      }
      auto curNodes = icb.candidateSIMDChains[chain].bottomLanes;
      std::set<size_t> prevNodes;
      for (int i = 0; i < icb.candidateSIMDChains[chain].length; i++) {
	auto& sets = chainPosToPackVecMap[chain][i];

        auto res = returnBestSet(chainPosToPackVecMap[chain][i], simdChainSets,
                                  icb, chainPosToPackVecMap, curNodes,
                                  {prevNodes.begin(), prevNodes.end()});
	size_t vecCost = res.cost;
	if (i != 0) {
	  // cost of arith for prevNodes vector
	  candidateCost += ci->vecArithCost;
	}

        selectedSetsForCand.insert(selectedSetsForCand.end(),
                                   res.selectedSets.begin(),
                                   res.selectedSets.end());
        candidateCost += res.cost;
	candidateSubtrees.insert(candidateSubtrees.end(), res.subtrees.begin(), res.subtrees.end());

        prevNodes = {curNodes.begin(), curNodes.end()};
        for (int j = 0; j < curNodes.size(); j++) {
          curNodes[j] = icb.childParentMap[curNodes[j]];
        }
      }
    }

    // First: Node id, Second: Vector Width
    std::set<std::pair<size_t, size_t>> coveredChildrenWithSize;
    for (auto &chain : simdChainSets[cand].SIMDChains) {
      auto curNodes = icb.candidateSIMDChains[chain].bottomLanes;
      for (int i = 1; i < icb.candidateSIMDChains[chain].length; i++) {
        for (int j = 0; j < curNodes.size(); j++) {
          curNodes[j] = icb.childParentMap[curNodes[j]];
        }
      }
      for (auto& n : curNodes) {
	coveredChildrenWithSize.insert({n, curNodes.size()});
      }
    }

    if (source.size() < 2) {
      // This is the original call to returnBestSet, so we need to account for
      // all nodes not handled by the selected sets.
      std::unordered_map<std::string, size_t> childrenStrings;
      for (auto& child : coveredChildrenWithSize) {
	childrenStrings.insert({icb.nodes[child.first]->id(), child.second});
      }
      candidateCost += scc.getCost(icb.nodes[source[0]], &childrenStrings);
      
    } else {
      // handle inputs to _source_ that were not covered by the chains of cand
      std::set<size_t> coveredChildren;
      for (auto& c : coveredChildrenWithSize)
	coveredChildren.insert(c.first);
      if (pruned.size() > 0) {
        coveredChildren.insert(pruned.begin(), pruned.end());
        // cost of arithmetic for the pruned (i.e. preselected) input vector
        candidateCost += ci->vecArithCost;
      }
      std::multiset<size_t, std::greater<size_t>> inputHistos;
      std::multiset<size_t> inputGaussians;
      for (int i = 0; i < source.size(); i++) {
        inputHistos.insert(icb.parentChildrenHistoMap[source[i]].size());
        inputGaussians.insert(icb.parentChildrenGaussMap[source[i]].size());
      }
      candidateCost += ci->histogramCost(inputHistos);
      candidateCost += ci->gaussCost(inputGaussians);
      
      for (auto &lane : source) {
        std::vector<size_t> serNodes;

        auto &laneChildren = icb.parentChildrenMap[lane];

        std::set_difference(laneChildren.begin(), laneChildren.end(),
                            coveredChildren.begin(), coveredChildren.end(),
                            std::inserter(serNodes, serNodes.begin()));
        if (serNodes.size() == 0)
          continue;
        for (auto &serNode : serNodes) {
	  auto serNodeRes = getVectorizationRec(serNode, icb);
          candidateCost += serNodeRes.second;
	  candidateSubtrees.push_back(serNodeRes.first);
        }
	// No. of arithemtic operations to combine the scalar inputs
	candidateCost += (serNodes.size()-1)*ci->scalarArithCost;
        // cost of inserting serial inputs of _lane_ into vector
        candidateCost += ci->getInsertCost(source.size());
      }
    }
    if (candidateCost < bestCandidateCost) {
      bestCandidateCost = candidateCost;
      bestSelectedSets = selectedSetsForCand;
      bestSubTrees = candidateSubtrees;
      bestSelectedSets.push_back(cand);
    }
  }
  return {bestSelectedSets, bestCandidateCost, bestSubTrees};
}

std::vector<std::pair<size_t, int>> PackingHeuristic::orderByPotential(std::vector<SIMDChain>& chains, std::vector<size_t> toOrder, InitialChainBuilder& icb, bool first) {

    std::vector<std::pair<size_t, int>> savings;
    for (auto& idx : toOrder) {
      auto& cand = chains[idx];
      // TODO: This should also account for the # of actual leafs below this chain
      int saving;
      if (icb.parentChildrenHistoMap.size() != 0) {
        saving = cand.length * (cand.bottomLanes.size() - 1) -
                 ci->getInsertCost(cand.bottomLanes.size()) *
                     (cand.bottomLanes.size() - cand.gatherCount) -
                 ci->getHistogramPenalty(cand.gatherCount);
	if (first)
	  saving -= cand.bottomLanes.size()*ci->getExtractCost(cand.bottomLanes.size());
      } else {
        std::multiset<size_t> inputGaussians;
	size_t gaussCount = 0;
        for (auto& lane : cand.bottomLanes) {
	  size_t gaussians = icb.parentChildrenGaussMap[lane].size();
	  gaussCount += gaussians;
	  inputGaussians.insert(gaussians);
        }
	// calc serial cost and vector cost and calc difference
	size_t vecCost = ci->gaussCost(inputGaussians);
	size_t serCost = gaussCount*(ci->gaussArithCost + ci->scalarArithCost);
	saving = serCost - vecCost;
      }
      savings.push_back({idx, saving});
    }

    std::sort(savings.begin(), savings.end(),
              [](auto &a, auto &b) { return a.second > b.second; });
    return savings;


}

std::vector<SIMDChainSet>
PackingHeuristic::buildSIMDChainSets(std::vector<size_t> &candidateSIMDChains,
                                     bool first, InitialChainBuilder &icb,
                                     size_t setsGoal) {
  auto savings = orderByPotential(icb.candidateSIMDChains, candidateSIMDChains, icb, first);
  std::unordered_map<size_t, std::vector<size_t>> simdChainNodeConflicts;
  for (auto &simdChain : candidateSIMDChains) {
    auto &chain = icb.candidateSIMDChains[simdChain];
    for (int k = 0; k < chain.bottomLanes.size(); k++) {
      size_t cur = chain.bottomLanes[k];
      for (int j = 0; j < chain.length; j++) {
        simdChainNodeConflicts[cur].push_back(simdChain);
        cur = icb.childParentMap[cur];
      }
    }
  }

  std::unordered_map<size_t, std::set<size_t>> simdChainConflicts;
  for (auto &simdChain : candidateSIMDChains) {
    auto &chain = icb.candidateSIMDChains[simdChain];
    for (int k = 0; k < chain.bottomLanes.size(); k++) {
      size_t cur = chain.bottomLanes[k];
      for (int j = 0; j < chain.length; j++) {
        auto &nodeConflicts = simdChainNodeConflicts[cur];
        simdChainConflicts[simdChain].insert(nodeConflicts.begin(),
                                             nodeConflicts.end());
        cur = icb.childParentMap[cur];
      }
    }
  }
  // If this not the first chain packing, there can be no dependencies,
  // because because all SIMDChains lead to the same vector
  if (first) {
    for (int i = 0; i < icb.candidateSIMDChains.size(); i++) {

      for (int j = i + 1; j < icb.candidateSIMDChains.size(); j++) {
        bool dependency = false;
        auto &chaini = icb.candidateSIMDChains[i];

        auto &chainj = icb.candidateSIMDChains[j];

        std::vector<size_t> headsi;
        for (auto &bot : chaini.bottomLanes) {
          size_t cur = bot;
          for (int k = 1; k < chaini.length; k++) {
            cur = icb.childParentMap[cur];
          }
          headsi.push_back(cur);
        }

        std::vector<size_t> headsj;
        for (auto &bot : chainj.bottomLanes) {
          size_t cur = bot;
          for (int k = 1; k < chainj.length; k++) {
            cur = icb.childParentMap[cur];
          }
          headsj.push_back(cur);
        }

        for (auto &headi : headsi) {
          for (auto &headj : headsj) {
            if (icb.dependsOn[headi].find(headj) !=
                    icb.dependsOn[headi].end() ||
                icb.dependsOn[headj].find(headi) !=
                    icb.dependsOn[headj].end()) {
              dependency = true;
              break;
            }
          }
          if (dependency)
            break;
        }

        if (dependency) {
          simdChainConflicts[i].insert(j);
          simdChainConflicts[j].insert(i);
        }
      }
    }
  }
  std::vector<SIMDChainSet> SIMDChainSets;
  for (int i = 0; i < std::min(setsGoal, savings.size()); i++) {
    SIMDChainSet scs;
    scs.originatingChain = -1;
    size_t initialSIMDChain = savings[i].first;
    std::vector<std::pair<size_t, int>> ordered = {{initialSIMDChain, 0}};
    std::vector<size_t> avail = candidateSIMDChains;
    // For now only take best option according to ordering
    while (true) {
      scs.SIMDChains.push_back(ordered[0].first);
      auto &conf = simdChainConflicts[ordered[0].first];

      std::vector<size_t> newAvail;

      std::set_difference(avail.begin(), avail.end(), conf.begin(), conf.end(),
                          std::inserter(newAvail, newAvail.begin()));
      if (newAvail.size() == 0)
        break;
      avail = newAvail;
      ordered = orderByPotential(icb.candidateSIMDChains, newAvail, icb, first);
    }
    SIMDChainSets.push_back(scs);
  }
  return SIMDChainSets;
}

std::pair<vectorizationResultInfo, size_t> PackingHeuristic::getVectorizationRec(size_t rootNode, InitialChainBuilder& icb) {
  if (subTreeCache.find(rootNode) != subTreeCache.end())
    return subTreeCache[rootNode];
  size_t treeSize = icb.dependsOn[rootNode].size();
  if (rootNode == 0) {
    overallTreeSize = treeSize;
  }
  if (treeSize < 2) {
    SerialCostCalc scc(ci.get());
    auto cost = scc.getCost(icb.nodes[rootNode]);
    return {{}, cost};
  }

  std::set<size_t> allChildChains;
  std::queue<size_t> nodesToHandle;
  nodesToHandle.push(rootNode);

  // calculate (indirect) children chains of rootNode
  while (!nodesToHandle.empty()) {
    auto node = nodesToHandle.front();
    allChildChains.insert(icb.childChains[node].begin(), icb.childChains[node].end());
    for (auto& childNode : icb.parentChildrenMap[node])
      nodesToHandle.push(childNode);
    nodesToHandle.pop();
  }

  size_t usedCandGoal =  spnc::option::chainCandidates.get(*conf);
  // scale no. of simd chains to produce according to treeSize
  size_t candGoal = ((double)usedCandGoal)*((double) treeSize/(double) overallTreeSize)+1;
  
  size_t firstSIMDChainIdx = icb.candidateSIMDChains.size();
  std::vector<size_t> scalarProductChains;
  std::set_intersection(allChildChains.begin(), allChildChains.end(),
                        icb.scalarProductChain.begin(),
                        icb.scalarProductChain.end(),
                        std::back_inserter(scalarProductChains));

  icb.generateCandidateChains(scalarProductChains, icb.productChainConflicts, candGoal);


  std::vector<size_t> scalarSumChains;
  std::set_intersection(allChildChains.begin(), allChildChains.end(),
                        icb.scalarSumChain.begin(),
                        icb.scalarSumChain.end(),
                        std::back_inserter(scalarSumChains));

  icb.generateCandidateChains(scalarSumChains, icb.sumChainConflicts, candGoal);

  std::vector<size_t> scalarWeightedSumChains;
  std::set_intersection(allChildChains.begin(), allChildChains.end(),
                        icb.scalarWeightedSumChain.begin(),
                        icb.scalarWeightedSumChain.end(),
                        std::back_inserter(scalarWeightedSumChains));

  icb.generateCandidateChains(scalarWeightedSumChains, icb.weightedSumChainConflicts, candGoal);

  std::vector<size_t> availableSIMDChains(icb.candidateSIMDChains.size() -
                                          firstSIMDChainIdx);

  for (int i = firstSIMDChainIdx; i < icb.candidateSIMDChains.size(); i++) {
    availableSIMDChains[i - firstSIMDChainIdx] = i;
  }

  size_t usedInit = spnc::option::rootCand.get(*conf);
  
  auto simdChainSets =
      buildSIMDChainSets(availableSIMDChains, true, icb, usedInit);
  std::vector<size_t> initialChainSets;
  for (int i = 0; i < simdChainSets.size(); i++) {
    initialChainSets.push_back(i);
  }

  std::queue<size_t> selectedChains;
  for (auto &set : simdChainSets) {
    for (auto &usedChain : set.SIMDChains) {
      selectedChains.push(usedChain);
    }
  }
  // return indexes of all SIMDChainSets that originated from a specific vector
  // in a specific chain
  std::unordered_map<size_t, std::unordered_map<size_t, std::vector<size_t>>>
      chainPosToPackVecMap;
  std::unordered_set<size_t> handledChains;
  while (!selectedChains.empty()) {
    size_t chainIdx = selectedChains.front();
    selectedChains.pop();
    if (handledChains.find(chainIdx) != handledChains.end())
      continue;

    handledChains.insert(chainIdx);
    auto &chain = icb.candidateSIMDChains[chainIdx];
    std::vector<size_t> currentVec = chain.bottomLanes;
    std::vector<size_t> prevVec(currentVec.size());
    for (int j = 0; j < chain.length; j++) {
      // For each vector in chain, try different ways to produce the other
      // inputs necessary for that vector, each possibility is a SIMDChainSet

      std::set<size_t> childScalarChains;
      std::unordered_map<size_t, std::set<size_t>> conflictMap;
      for (int k = 0; k < currentVec.size(); k++) {
        std::set<size_t> conflicts;
        auto &lane = currentVec[k];

        for (auto &childChain : icb.childChains[lane]) {
          // If the chain leading to the node _lane_ goes through the child of
          // lane that is part of _chain_, we can not select it
          if (j != 0 && icb.scalarChains[childChain][0] == prevVec[k])
            continue;

          // This chain must have conflicts with all other chains that are
          // also leading to _lane_ such that the possible vectors are always
          // direct input vectors for the vector "chain[j]"
          conflicts.insert(childChain);
          childScalarChains.insert(childChain);
        }

        for (auto &confChain : conflicts) {
          conflictMap[confChain] = conflicts;
        }
        prevVec[k] = lane;
        currentVec[k] = icb.childParentMap[lane];
      }
      size_t firstSIMDChainIdx = icb.candidateSIMDChains.size();
      std::vector<size_t> childScalarChainsVec(childScalarChains.begin(), childScalarChains.end());
      double avgChildCount = 0;
      for (auto& lane : prevVec) {
	avgChildCount+= icb.parentChildrenMap[lane].size();
      }
      avgChildCount /= prevVec.size();
      if (j != 0)
	avgChildCount -= 1.0;

      size_t usedFactor = spnc::option::depChains.get(*conf);
      icb.generateCandidateChains(childScalarChainsVec, conflictMap, avgChildCount*usedFactor);
      std::vector<size_t> generatedSIMDChains(icb.candidateSIMDChains.size() -
                                              firstSIMDChainIdx);
      for (int i = 0; i < icb.candidateSIMDChains.size() - firstSIMDChainIdx;
           i++) {
        generatedSIMDChains[i] = firstSIMDChainIdx + i;
      }

      size_t usedDepCount = spnc::option::depCand.get(*conf);
      
      auto newSIMDChainSets = buildSIMDChainSets(generatedSIMDChains, false,
                                                 icb, usedDepCount);

      for (auto &set : newSIMDChainSets) {
        size_t setId = simdChainSets.size();
        set.originatingChain = chainIdx;
        set.posInOriginatingChain = j;
        simdChainSets.push_back(set);
        chainPosToPackVecMap[chainIdx][j].push_back(setId);
        for (auto &usedChain : set.SIMDChains) {
          selectedChains.push(usedChain);
        }
      }
    }
  }

  auto res = returnBestSet(initialChainSets, simdChainSets, icb,
                           chainPosToPackVecMap, {rootNode}, {});

  std::unordered_map<size_t, std::unordered_map<size_t, size_t>>
      chainPosToVecId;
  vectorizationResultInfo vecRes;
  for (auto &set : res.selectedSets) {
    for (auto &chainIdx : simdChainSets[set].SIMDChains) {
      auto &chain = icb.candidateSIMDChains[chainIdx];
      auto curVec = chain.bottomLanes;
      for (int i = 0; i < chain.length; i++) {
        std::vector<NodeReference> vector;
        for (auto &lane : curVec) {
          vector.push_back(icb.nodes[lane]);
        }
        size_t vecId = vecRes.vectors.size();
        vecRes.vectors.push_back(vector);

        for (auto &ref : vector) {
          vecRes.partOf.insert({ref->id(), vecId});
        }
        if (i != 0) {
          vecRes.directVecInputs[vecId].insert(vecId - 1);
        }
        chainPosToVecId[chainIdx][i] = vecId;
        for (int j = 0; j < curVec.size(); j++) {
          curVec[j] = icb.childParentMap[curVec[j]];
        }
      }
    }
  }

  for (auto &setIdx : res.selectedSets) {
    auto &set = simdChainSets[setIdx];
    if (set.originatingChain == -1) {
      // This is the top level set, thus its chains cannot be a direct input
      // to other vectors
      continue;
    }
    auto sourceIt =
        chainPosToVecId[set.originatingChain].find(set.posInOriginatingChain);
    assert(sourceIt != chainPosToVecId[set.originatingChain].end());
    for (auto &chainIdx : set.SIMDChains) {
      auto &chain = icb.candidateSIMDChains[chainIdx];

      auto inputIt = chainPosToVecId[chainIdx].find(chain.length - 1);
      assert(inputIt != chainPosToVecId[chainIdx].end());
      vecRes.directVecInputs[sourceIt->second].insert(inputIt->second);
    }
  }
  for (auto& subtree : res.subtrees) {
    std::unordered_map<size_t, size_t> vecIdMap;
    for (int i = 0; i < subtree.vectors.size(); i++) {
      auto newId = vecRes.vectors.size();
      vecRes.vectors.push_back(subtree.vectors[i]);
      vecIdMap.insert({i, newId});
    }

    for (auto& node : subtree.partOf) {
      vecRes.partOf.insert({node.first, vecIdMap[node.second]});
    }

    for (auto& vec : subtree.directVecInputs) {
      auto& newInputs = vecRes.directVecInputs[vecIdMap[vec.first]];

      for (auto& oldInput : vec.second) {
	newInputs.insert(vecIdMap[oldInput]);
      }
    }
  }

  
  subTreeCache.insert({rootNode, {vecRes, res.cost}});
  return {vecRes, res.cost};
}

vectorizationResultInfo PackingHeuristic::getVectorization(IRGraph &graph, size_t width, const Configuration& config) {
  std::cout << "run heuristic" << std::endl;
  conf = &config;
  ci = std::make_unique<CostInfo>(width);
  InitialChainBuilder icb(width);
  icb.performInitialBuild(graph.rootNode());
  return getVectorizationRec(0, icb).first;
}
