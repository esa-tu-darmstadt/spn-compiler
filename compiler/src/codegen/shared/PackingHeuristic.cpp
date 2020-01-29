#include "PackingHeuristic.h"
#include "HeuristicChainBuilder.h"
#include "InitialChainBuilder.h"
#include <iostream>
#include <queue>

#define USE_INIT 1
#define INSERTION_COST 2
#define USED_CANDIDATES_INIT 20
#define USED_CANDIDATES_DEP 5

struct SIMDChainSet {
  std::vector<size_t> SIMDChains;
  int originatingChain;
  int posInOriginatingChain;
};


std::vector<std::pair<size_t, int>> orderByPotential(std::vector<SIMDChain>& chains, std::vector<size_t> toOrder) {

    std::vector<std::pair<size_t, int>> savings;
    for (auto& idx : toOrder) {
      auto& cand = chains[idx];
      // TODO: This should also account for the # of actual gather loads
      size_t saving = cand.length*(cand.bottomLanes.size()-1)-INSERTION_COST*(cand.bottomLanes.size()-cand.gatherCount);
      savings.push_back({idx, saving});
    }

    std::sort(savings.begin(), savings.end(),
              [](auto &a, auto &b) { return a.second > b.second; });
    return savings;


}

std::vector<SIMDChainSet> buildSIMDChainSets(std::vector<size_t>& candidateSIMDChains, bool first, InitialChainBuilder& icb, size_t setsGoal) {
    auto savings = orderByPotential(icb.candidateSIMDChains, candidateSIMDChains);
    std::unordered_map<size_t, std::vector<size_t>> simdChainNodeConflicts;
    for (auto& simdChain : candidateSIMDChains) {
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
    for (auto& simdChain : candidateSIMDChains) {
      auto &chain = icb.candidateSIMDChains[simdChain];
      for (int k = 0; k < chain.bottomLanes.size(); k++) {
        size_t cur = chain.bottomLanes[k];
        for (int j = 0; j < chain.length; j++) {
	  auto& nodeConflicts = simdChainNodeConflicts[cur];
          simdChainConflicts[simdChain].insert(nodeConflicts.begin(), nodeConflicts.end());
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

        std::set_difference(avail.begin(), avail.end(), conf.begin(),
                            conf.end(),
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
  std::cout << "test" << std::endl;
  if (USE_INIT) {
    InitialChainBuilder icb(width);
    icb.performInitialBuild(graph.rootNode);

    std::vector<size_t> availableSIMDChains(icb.candidateSIMDChains.size());

    for (int i = 0; i < icb.candidateSIMDChains.size(); i++) {
      availableSIMDChains[i] = i;
    }
    auto simdChainSets = buildSIMDChainSets(availableSIMDChains, true, icb,
                                            USED_CANDIDATES_INIT);

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
      std::vector<size_t> prevVec(currentVec.size());;
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
      std::cout << "set from " << chainset.originatingChain << " pos "
                << chainset.posInOriginatingChain << " consitituents "
                << std::endl;
      for (auto& simdchain : chainset.SIMDChains) {
	std::cout << simdchain << ",";
      }
      std::cout << std::endl;
    }

    // TODO: Branch and cut when evaluating different options for packs for
    // (chain, pos) also save best config for already handled chain, so that
    // calc'ing chainSet cost is after some time just adding up chain cost
  }
  else {
    HeuristicChainBuilder hcb(graph, width);
  }
  return {};
  
}
