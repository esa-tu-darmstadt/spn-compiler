#include "InitialChainBuilder.h"
#include <iostream>
#define CANDIDATES_GOAL 500
template <typename T> class vecSetter : public BaseVisitor {
public:
  vecSetter(T &Sum_, T &WeightedSum_, T &Product_ )
      : Sum(Sum_), WeightedSum(WeightedSum_), Product(Product_)

  {}

  void visitProduct(Product &n, arg_t arg) {cur = &Product; }

  void visitSum(Sum &n, arg_t arg) {cur = &Sum; }

  void visitWeightedSum(WeightedSum &n, arg_t arg) { cur = &WeightedSum; }

  T &Sum;
  T &WeightedSum;
  T &Product;
  T*cur;
};

InitialChainBuilder::InitialChainBuilder(size_t width_)
  : width(width_) {}

void InitialChainBuilder::performInitialBuild(NodeReference root) {
  nodes.push_back(root);
  root->accept(*this, std::make_shared<std::vector<size_t>>());

  for (auto& histo : histograms) {
    vecSetter vs(scalarSumChain, scalarWeightedSumChain, scalarProductChain);
    auto it = childParentMap.find(histo);
    std::vector<size_t> nodeSeq;
    nodeSeq.push_back(it->second);
    it = childParentMap.find(it->second);
    size_t height = 2;
    while (it != childParentMap.end()) {
      nodeSeq.push_back(it->second);
      nodes[it->second]->accept(vs, {});
      vs.cur->push_back(std::vector<size_t>(nodeSeq.rbegin(), nodeSeq.rend()));
      height++;
      it = childParentMap.find(it->second);
    }
  }

  findConflicts(scalarSumChain, sumConflicts);
  findConflicts(scalarWeightedSumChain, weightedSumConflicts);
  findConflicts(scalarProductChain, productConflicts);

  
  findChainConflicts(scalarSumChain, sumConflicts, sumChainConflicts);
  findChainConflicts(scalarWeightedSumChain, weightedSumConflicts,weightedSumChainConflicts);
  findChainConflicts(scalarProductChain, productConflicts,productChainConflicts);
  /*
  for (int i = 0; i < scalarWeightedSumChain.size(); i++) {
    auto& c  = scalarWeightedSumChain[i];
    std::cout << "chain " << i << std::endl;
    for (auto node : c) {
      std::cout << nodes[node]->id() << ",";
    }
    std::cout << std::endl << "conflicts " << std::endl;
    for (auto& conf : weightedSumChainConflicts[i]) {
      std::cout << conf << ",";
    }
    std::cout << std::endl;
  }
  */
  generateCandidateChains(scalarProductChain, productChainConflicts, CANDIDATES_GOAL);
  generateCandidateChains(scalarSumChain, sumChainConflicts, CANDIDATES_GOAL);
  generateCandidateChains(scalarWeightedSumChain, weightedSumChainConflicts, CANDIDATES_GOAL);
  /*
  for (auto& simdChain : candidateSIMDChains) {
    std::cout << "new simd chain, length:" << simdChain.length << " roots " << std::endl;
    for (auto& bot : simdChain.bottomLanes) {
      std::cout << nodes[bot]->id() << ",";
    }
    std::cout << std::endl;
  }
  */
}

size_t InitialChainBuilder::recChainGen(std::vector<size_t> selected, std::vector<size_t> stillAvailable, std::vector<std::vector<size_t>>& chains, std::unordered_map<size_t, std::set<size_t>>& conflicts) {
  int commonLength = -1;

  if (selected.size() > 0)
    commonLength = chains[selected[0]].size();
  
  if (selected.size() > 1) {
    // emit candidate chain
    for (int i = 1; i < selected.size(); i++) {
      commonLength = std::min(commonLength, (int) chains[selected[i]].size());
    }

    size_t gatherCount = 0;
    std::vector<size_t> bottomLanes;
    for (auto& chain: selected) {
      if (chains[chain].size() == commonLength) {
	gatherCount++;
	
      }
      bottomLanes.push_back(chains[chain][commonLength-1]);
    }
    candidateSIMDChains.push_back({gatherCount, static_cast<size_t>(commonLength), bottomLanes});
  }

  if (stillAvailable.size() == 0)
    return 0;

  if (selected.size() == width) {
    return 1;
  }

  // favor same length chains here (i.e. those that will enable gather load)
  std::vector<size_t> sortedAvail(stillAvailable.begin(), stillAvailable.end());
  std::sort(sortedAvail.begin(), sortedAvail.end(), [&](size_t a, size_t b) {
    if (commonLength != -1) {
      if (chains[a].size() == commonLength && chains[b].size() != commonLength)
        return true;
      else if (chains[b].size() == commonLength)
        return false;
    }
    return chains[a].size() > chains[b].size();
  });

  // another sort mechanism might be a score judging both the length of a
  // sequence and how different it is to others already selected, e.g. multiply
  // length with (1/(max(common prefix))
  size_t goalFullWidthChains =
      ((size_t)std::pow(candidatesGoal,
                        ((double)width - selected.size()) / ((double)width))) +
      1;
  size_t buildFullWidthChains = 0;
  auto it = sortedAvail.begin();
  std::vector<size_t> used;
  while (buildFullWidthChains < goalFullWidthChains && it != sortedAvail.end()) {
    // find next scalarChain to select
    size_t maxScore = 0;
    auto tempIt = it;
    for (; tempIt != sortedAvail.end() && chains[*tempIt].size() > maxScore; tempIt++) {
      // determine maximum common prefix between chain tempIt and already emitted chains
      auto& candidateChain = chains[*tempIt];
      size_t maxCommonPrefix = 0;
      for (auto& handledChainIdx : used) {
	auto& handledChain = chains[handledChainIdx];
	size_t commonPrefix = 0;
	for (int i = 0; i < std::min(candidateChain.size(), handledChain.size()); i++) {
	  if (candidateChain[i] != handledChain[i])
	    break;
	  commonPrefix++;
	}
	maxCommonPrefix = std::max(maxCommonPrefix, commonPrefix);
      }
      // TODO: Heuristic, maybe change how we incorporate maxCommonPrefix into score
      if (maxScore < candidateChain.size() - maxCommonPrefix) {
	maxScore = candidateChain.size() - maxCommonPrefix;
        it = tempIt;
      }
    }
    if (maxScore == 0)
      break;

    used.push_back(*it);
    std::vector<size_t> newSel(selected);
    newSel.push_back(*it);
    std::vector<size_t> newAvail;
    auto& conf = conflicts[*it];
    
    std::set_difference(stillAvailable.begin(),
        stillAvailable.end(), conf.begin(), conf.end(),
        std::inserter(newAvail, newAvail.begin()));

    it++;
    buildFullWidthChains += recChainGen(newSel, newAvail, chains, conflicts);
  }
  return buildFullWidthChains;
}

void InitialChainBuilder::generateCandidateChains( std::vector<std::vector<size_t>>& chains, std::unordered_map<size_t, std::set<size_t>>& conflicts, size_t chainsGoal) {
  candidatesGoal = chainsGoal;
  std::vector<size_t> availableChains(chains.size());
  for (size_t i = 0; i < chains.size(); i++) {
    availableChains[i] = i;
  }
  recChainGen({}, availableChains, chains, conflicts);
}
   
void InitialChainBuilder::findChainConflicts(
    std::vector<std::vector<size_t>> &chains,
    std::unordered_map<size_t, std::vector<size_t>> &nodeConflicts,
    std::unordered_map<size_t, std::set<size_t>> &chainConflicts) {

  for (int id  = 0 ; id < chains.size(); id++) {
    auto& chain = chains[id]; 
    for (auto& node : chain) {
      auto& conflictsAtNode = nodeConflicts[node];
      chainConflicts[id].insert(conflictsAtNode.begin(), conflictsAtNode.end());
    }

    for (int id2 = id+1; id2 < chains.size(); id2++) {
      auto &chain2 = chains[id2];
      if (dependsOn[chain[0]].find(chain2[0]) != dependsOn[chain[0]].end() ||
          dependsOn[chain2[0]].find(chain[0]) != dependsOn[chain2[0]].end()) {
	chainConflicts[id].insert(id2);
	chainConflicts[id2].insert(id);
      }
    }
  }
}


void
InitialChainBuilder::findConflicts(std::vector<std::vector<size_t>> &chains,
				   std::unordered_map<size_t, std::vector<size_t>> &conflicts) {
  for (int id  = 0 ; id < chains.size(); id++) {
    auto& chain = chains[id];
    for (auto& node : chain) {
      conflicts[node].push_back(id);
    }
  }
}

void InitialChainBuilder::visitHistogram(Histogram &n, arg_t arg) {
  size_t ownId = nodes.size()-1;
  auto parentId = childParentMap[ownId];
  auto it = coveredPreLeafNodes.find(parentId);
  if (it != coveredPreLeafNodes.end())
    return;
  std::vector<size_t>* prevNodes = (std::vector<size_t>*) arg.get();
  for (auto& prev : *prevNodes) {
    dependsOnHistograms[prev].insert(ownId);
  }

  parentChildrenHistoMap[prevNodes->back()].push_back(ownId);
  histograms.push_back(ownId);
  coveredPreLeafNodes.insert(parentId);
}

void InitialChainBuilder::visitProduct(Product &n, arg_t arg) {

  size_t ownId = nodes.size()-1;
  std::vector<size_t>* prevNodes = (std::vector<size_t>*) arg.get();
  for (auto& prev : *prevNodes) {
    dependsOn[prev].insert(ownId);
  }

  if (prevNodes->size() > 0) {
    parentChildrenMap[prevNodes->back()].push_back(ownId);
  }
  auto newPrevs = std::make_shared<std::vector<size_t>>(*prevNodes);
  newPrevs->push_back(ownId);
  for (auto &c : *n.multiplicands()) {
    size_t childId = nodes.size();
    nodes.push_back(c);
    childParentMap.insert({childId, ownId});
    c->accept(*this, newPrevs);
  }
}

void InitialChainBuilder::visitSum(Sum &n, arg_t arg) {
  
  size_t ownId = nodes.size()-1;
  std::vector<size_t>* prevNodes = (std::vector<size_t>*) arg.get();
  for (auto& prev : *prevNodes) {
    dependsOn[prev].insert(ownId);
  }

  if (prevNodes->size() > 0) {
    parentChildrenMap[prevNodes->back()].push_back(ownId);
  }
  auto newPrevs = std::make_shared<std::vector<size_t>>(*prevNodes);
  newPrevs->push_back(ownId);
  for (auto &c : *n.addends()) {
    
    size_t childId = nodes.size();
    nodes.push_back(c);
    childParentMap.insert({childId, ownId});
    c->accept(*this, newPrevs);

  }
}

void InitialChainBuilder::visitWeightedSum(WeightedSum &n, arg_t arg) {
  size_t ownId = nodes.size()-1;
  std::vector<size_t>* prevNodes = (std::vector<size_t>*) arg.get();
  for (auto& prev : *prevNodes) {
    dependsOn[prev].insert(ownId);
  }

  if (prevNodes->size() > 0) {
    parentChildrenMap[prevNodes->back()].push_back(ownId);
  }
  
  auto newPrevs = std::make_shared<std::vector<size_t>>(*prevNodes);
  newPrevs->push_back(ownId);
  for (auto &c : *n.addends()) {
    size_t childId = nodes.size();
    nodes.push_back(c.addend);
    childParentMap.insert({childId, ownId});
    c.addend->accept(*this, newPrevs);
  }
}
