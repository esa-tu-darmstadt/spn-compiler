#include "PackingHeuristic.h"
#include "HeuristicChainBuilder.h"
#include <iostream>
#include <queue>

#define USE_INIT 1
#define USED_CANDIDATES_INIT 20
#define USED_CANDIDATES_DEP 5


class SerialCostCalc : public BaseVisitor {
public:
  SerialCostCalc(CostInfo *ci_) : ci(ci_) {}
  size_t getCost(const NodeReference &rootNode, std::unordered_set<std::string>* vectorized_) {
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

  void visitProduct(Product &n, arg_t arg) {
    for (auto &c : *n.multiplicands()) {
      if (vectorized && vectorized->find(c->id()) != vectorized->end()) {
	// extract and arith
	curCost += ci->extractCost + ci->scalarArithCost;
      } else {
        curCost += ci->scalarArithCost;
        c->accept(*this, {});
      }
    }
  }

  void visitSum(Sum &n, arg_t arg)  {
    for (auto &c : *n.addends()) {
      if (vectorized && vectorized->find(c->id()) != vectorized->end()) {
	// extract and arith
	curCost += ci->extractCost + ci->scalarArithCost;
      } else {
        curCost += ci->scalarArithCost;
        c->accept(*this, {});
      }
    }
  }

  void visitWeightedSum(WeightedSum &n, arg_t arg)  {
    for (auto &c : *n.addends()) {
      if (vectorized && vectorized->find(c.addend->id()) != vectorized->end()) {
	// extract and arith
	curCost += ci->extractCost + ci->scalarArithCost;
      } else {
        curCost += ci->scalarArithCost;
        c.addend->accept(*this, {});
      }
    }
  }

private:
  size_t curCost;
  std::unordered_set<std::string>* vectorized = nullptr;
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
  if (source.size() == 0) {
    bestCandidateCost = scc.getCost(icb.nodes[0]);
  } else {
    for (auto &lane : source) {
      std::vector<size_t> serNodes;

      auto &laneChildren = icb.parentChildrenMap[lane];
      std::set_difference(laneChildren.begin(), laneChildren.end(),
                          pruned.begin(), pruned.end(),
                          std::inserter(serNodes, serNodes.begin()));

      if (serNodes.size() > 0) {
        // cost of insertion
        bestCandidateCost += ci->insertCost;
        // cost of scalar arithmetic operations for producing lane
        bestCandidateCost += (serNodes.size() - 1)*ci->scalarArithCost;
        for (auto &child : serNodes) {
          bestCandidateCost += scc.getCost(icb.nodes[child]);
        }
      }
    }
    if (pruned.size() != 0) {
      // cost of mul/add of pruned vector to source vector
      bestCandidateCost += ci->vecArithCost;
    }
  }

  std::cout << "ser cost: " << bestCandidateCost << std::endl;
  
  std::vector<size_t> bestSelectedSets;

  for (auto& cand : candidates) {
    size_t candidateCost = 0;
    std::vector<size_t> selectedSetsForCand;
    for (auto& chain : simdChainSets[cand].SIMDChains) {
      if (source.size() != 0) {
        // cost of arith to source for vector produced by chain
        // if this is the initial call to returnBestSet (i.e. source.size() ==
        // 0), the cost of extracting the result of chain will be accounted for
        // in serialCostCalc
        candidateCost += ci->vecArithCost;
      }
      auto curNodes = icb.candidateSIMDChains[chain].bottomLanes;
      std::set<size_t> prevNodes;
      for (int i = 0; i < icb.candidateSIMDChains[chain].length; i++) {
        // first compute cost of doing all paths leading here serially
	std::cout << "evaluate chain " << chain << " pos " << i << std::endl;
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

        prevNodes = {curNodes.begin(), curNodes.end()};
        for (int j = 0; j < curNodes.size(); j++) {
          curNodes[j] = icb.childParentMap[curNodes[j]];
        }
      }
    }

    std::set<size_t> coveredChildren;
    for (auto &chain : simdChainSets[cand].SIMDChains) {
      auto curNodes = icb.candidateSIMDChains[chain].bottomLanes;
      for (int i = 1; i < icb.candidateSIMDChains[chain].length; i++) {
        for (int j = 0; j < curNodes.size(); j++) {
          curNodes[j] = icb.childParentMap[curNodes[j]];
        }
      }
      coveredChildren.insert(curNodes.begin(), curNodes.end());
    }

    if (source.size() == 0) {
      // This is the original call to returnBestSet, so we need to account for
      // all nodes not handled by the selected sets.
      std::unordered_set<std::string> childrenStrings;
      for (auto& child : coveredChildren) {
	childrenStrings.insert(icb.nodes[child]->id());
      }
      candidateCost += scc.getCost(icb.nodes[0], &childrenStrings);
      
    } else {
      // handle inputs to _source_ that were not covered by the chains of cand
      if (pruned.size() > 0) {
        coveredChildren.insert(pruned.begin(), pruned.end());
        // cost of arithmetic for the pruned (i.e. preselected) input vector
        candidateCost += ci->vecArithCost;
      }

      size_t gatherLoads = icb.parentChildrenHistoMap[source[0]].size();
      for (int i = 1; i < source.size(); i++) {
	gatherLoads = std::min(gatherLoads, icb.parentChildrenHistoMap[source[i]].size());
      }

      if (gatherLoads > 0) {
	// Cost of arithmetic operations for gathered input vectors
	candidateCost += gatherLoads*ci->vecArithCost;
      }
      for (auto &lane : source) {
        std::vector<size_t> serNodes;

        auto &laneChildren = icb.parentChildrenMap[lane];
        auto histoLoadCount = icb.parentChildrenHistoMap[lane].size()- gatherLoads;

        std::set_difference(laneChildren.begin(), laneChildren.end(),
                            coveredChildren.begin(), coveredChildren.end(),
                            std::inserter(serNodes, serNodes.begin()));
        if (serNodes.size() == 0 && histoLoadCount == 0)
          continue;
        for (auto &serNode : serNodes) {
          // TODO watchout for histograms here
          candidateCost += scc.getCost(icb.nodes[serNode]);
        }
	// No. of arithemtic operations to combine the scalar inputs
	candidateCost += (serNodes.size()+histoLoadCount-1)*ci->scalarArithCost;
        // cost of inserting serial inputs of _lane_ into vector
        candidateCost += ci->insertCost;
      }
    }
    std::cout << "cost for cand" << cand << " is " << candidateCost << std::endl;
    if (candidateCost < bestCandidateCost) {
      bestCandidateCost = candidateCost;
      bestSelectedSets = selectedSetsForCand;
      bestSelectedSets.push_back(cand);
    }
  }
  return {bestSelectedSets, bestCandidateCost};
}

std::vector<std::pair<size_t, int>> PackingHeuristic::orderByPotential(std::vector<SIMDChain>& chains, std::vector<size_t> toOrder) {

    std::vector<std::pair<size_t, int>> savings;
    for (auto& idx : toOrder) {
      auto& cand = chains[idx];
      // TODO: This should also account for the # of actual gather loads
      size_t saving = cand.length*(cand.bottomLanes.size()-1)-ci->insertCost*(cand.bottomLanes.size()-cand.gatherCount);
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
  auto savings = orderByPotential(icb.candidateSIMDChains, candidateSIMDChains);
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
      ordered = orderByPotential(icb.candidateSIMDChains, newAvail);
    }
    SIMDChainSets.push_back(scs);
  }
  return SIMDChainSets;
}

vectorizationResultInfo PackingHeuristic::getVectorization(IRGraph &graph, size_t width) {
  ci = std::make_unique<CostInfo>(width);
  if (USE_INIT) {
    InitialChainBuilder icb(width);
    icb.performInitialBuild(graph.rootNode);

    std::vector<size_t> availableSIMDChains(icb.candidateSIMDChains.size());

    for (int i = 0; i < icb.candidateSIMDChains.size(); i++) {
      availableSIMDChains[i] = i;
    }
    auto simdChainSets = buildSIMDChainSets(availableSIMDChains, true, icb,
                                            USED_CANDIDATES_INIT);
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
    // return indexes of all SIMDChainSets that originated from a specific vector in a specific chain
    std::unordered_map<size_t, std::unordered_map<size_t, std::vector<size_t>>> chainPosToPackVecMap;
    std::unordered_set<size_t> handledChains;
    while (!selectedChains.empty()) {
      size_t chainIdx = selectedChains.front();
      selectedChains.pop();
      if (handledChains.find(chainIdx) != handledChains.end())
	continue;
      
      handledChains.insert(chainIdx);
      auto& chain = icb.candidateSIMDChains[chainIdx];
      std::vector<size_t> currentVec = chain.bottomLanes;
      std::vector<size_t> prevVec(currentVec.size());
      for (int j = 0; j < chain.length; j++) {
        // For each vector in chain, try different ways to produce the other
        // inputs necessary for that vector, each possibility is a SIMDChainSet

        std::vector<std::vector<size_t>> scalarChains;
	std::unordered_map<size_t, std::set<size_t>> conflictMap;
        for (int k = 0; k < currentVec.size(); k++) {
	  std::set<size_t> conflicts;
	  auto& lane = currentVec[k];
	  for (auto& histo : icb.dependsOnHistograms[lane]) {
	    size_t cur = icb.childParentMap[histo];
	    std::vector<size_t> newScalarChain;
	    while (cur != lane) {
              newScalarChain.push_back(cur);
              cur = icb.childParentMap[cur];
            }
            if (newScalarChain.size() != 0 &&
                (j == 0 || newScalarChain.back() != prevVec[k])) {
              conflicts.insert(scalarChains.size());
              scalarChains.push_back(newScalarChain);
            }
          }

	  for (auto& confChain : conflicts) {
	    conflictMap[confChain] = conflicts;
	  }
	  prevVec[k] = lane;
          currentVec[k] = icb.childParentMap[lane];
        }
        size_t firstSIMDChainIdx = icb.candidateSIMDChains.size();
        icb.generateCandidateChains(scalarChains, conflictMap, 10);
	std::vector<size_t> generatedSIMDChains(icb.candidateSIMDChains.size()- firstSIMDChainIdx);
	for (int i = 0; i < icb.candidateSIMDChains.size()- firstSIMDChainIdx; i++) {
	  generatedSIMDChains[i] = firstSIMDChainIdx+i;
	}

	auto newSIMDChainSets = buildSIMDChainSets(generatedSIMDChains, false, icb, USED_CANDIDATES_DEP);

	for (auto& set : newSIMDChainSets) {
	  size_t setId = simdChainSets.size();
	  set.originatingChain = chainIdx;
	  set.posInOriginatingChain = j;
	  simdChainSets.push_back(set);
	  chainPosToPackVecMap[chainIdx][j].push_back(setId);
	  for (auto& usedChain : set.SIMDChains) {
	    selectedChains.push(usedChain);
	  }
	}
        // TODO also generate independent trees that produce an input for (j,k),
        // i.e. call PackingHeuristic recursively, and then make it possible for
        // such a tree to be part of a SIMDChainSet
      }
    }

    for (int i = 0; i< icb.candidateSIMDChains.size(); i++) {
      auto& simdchain = icb.candidateSIMDChains[i];
      std::cout << "chain id " << i << " length " << simdchain.length << "bottoms " << std::endl;
      for (auto& bot : simdchain.bottomLanes) {
	std::cout << icb.nodes[bot]->id() << ",";
      }
      std::cout << std::endl;
    }

    for (int i = 0; i < simdChainSets.size(); i++) {
      auto& chainset = simdChainSets[i];
      std::cout << "set id" << i << " from " << chainset.originatingChain << " pos "
                << chainset.posInOriginatingChain << " consitituents "
                << std::endl;
      for (auto& simdchain : chainset.SIMDChains) {
	std::cout << simdchain << ",";
      }
      std::cout << std::endl;
    }

    auto res = returnBestSet(initialChainSets, simdChainSets, icb, chainPosToPackVecMap, {}, {});
    std::cout << "selected :  " << std::endl;
    for (auto& set : res.selectedSets) {
      std::cout << set << "," ;
    }
    std::cout << std::endl;
    std::unordered_map<size_t, std::unordered_map<size_t, size_t>> chainPosToVecId;
    vectorizationResultInfo vecRes;
    for (auto& set : res.selectedSets) {
      for (auto& chainIdx : simdChainSets[set].SIMDChains) {
	auto& chain = icb.candidateSIMDChains[chainIdx];
	auto curVec = chain.bottomLanes;
	for (int i = 0; i < chain.length; i++) {
	  std::vector<NodeReference> vector;
	  for (auto& lane : curVec) {
	    vector.push_back(icb.nodes[lane]);
	  }
	  size_t vecId = vecRes.vectors.size();
	  vecRes.vectors.push_back(vector);

	  for (auto& ref : vector) {
	    vecRes.partOf.insert({ref->id(), vecId});
	  }
	  if (i != 0) {
	    vecRes.directVecInputs[vecId].insert(vecId-1);
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
    return vecRes;
  }
  else {
    HeuristicChainBuilder hcb(graph, width);
  }
  return {};
  
}
