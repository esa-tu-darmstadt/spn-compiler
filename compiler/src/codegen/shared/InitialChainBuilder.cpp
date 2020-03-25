#include "InitialChainBuilder.h"
#include <iostream>
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

  for (auto& leaf : leafs) {
    vecSetter vs(scalarSumChain, scalarWeightedSumChain, scalarProductChain);
    auto it = childParentMap.find(leaf);
    std::vector<size_t> nodeSeq;
    while (it != childParentMap.end()) {
      nodeSeq.push_back(it->second);
      size_t scalarChainId = scalarChains.size();
      scalarChains.push_back(std::vector<size_t>(nodeSeq.rbegin(), nodeSeq.rend()));
      nodes[it->second]->accept(vs, {});
      vs.cur->push_back(scalarChainId);
      it = childParentMap.find(it->second);
      if (it != childParentMap.end()) {
	childChains[it->second].push_back(scalarChainId);
      }
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
}

size_t InitialChainBuilder::recChainGen(
    std::vector<size_t> selected, std::vector<size_t> stillAvailable,
    std::unordered_map<size_t, std::set<size_t>> &conflicts) {
  int commonLength = -1;

  if (selected.size() > 0)
    commonLength = scalarChains[selected[0]].size();
  
  if (selected.size() > 1) {
    // emit candidate chain
    for (int i = 1; i < selected.size(); i++) {
      commonLength = std::min(commonLength, (int) scalarChains[selected[i]].size());
    }

    size_t gatherCount = 0;
    std::vector<size_t> bottomLanes;
    for (auto& chain: selected) {
      if (scalarChains[chain].size() == commonLength) {
	gatherCount++;
      }
      bottomLanes.push_back(scalarChains[chain][commonLength-1]);
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
      if (scalarChains[a].size() == commonLength && scalarChains[b].size() != commonLength)
        return true;
      else if (scalarChains[b].size() == commonLength)
        return false;
    }
    return scalarChains[a].size() > scalarChains[b].size();
  });

  size_t buildFullWidthChains = 0;
  auto it = sortedAvail.begin();
  std::vector<size_t> used;
  int i = 0;
  while (it != sortedAvail.end() &&
         ((selected.size() == 0 && buildFullWidthChains < candidatesGoal) ||
          (selected.size() != 0 && i < (width - selected.size() + 1)))) {
    // find next scalarChain to select
    size_t maxScore = 0;
    auto tempIt = it;
    for (; tempIt != sortedAvail.end() && scalarChains[*tempIt].size() > maxScore; tempIt++) {
      // determine maximum common prefix between chain tempIt and already emitted chains
      auto& candidateChain = scalarChains[*tempIt];
      size_t candidateLength = candidateChain.size();
      if (commonLength != -1)
	candidateLength = std::min(candidateLength, (unsigned long) commonLength);
      size_t maxCommonPrefix = 0;
      for (auto& handledChainIdx : used) {
	auto& handledChain = scalarChains[handledChainIdx];
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

    std::set<size_t> remainingAvailable(it, sortedAvail.end());
    
    std::set_difference(remainingAvailable.begin(),
        remainingAvailable.end(), conf.begin(), conf.end(),
        std::inserter(newAvail, newAvail.begin()));

    it++;
    i++;
    buildFullWidthChains += recChainGen(newSel, newAvail, conflicts);
  }
  return buildFullWidthChains;
}

void InitialChainBuilder::generateCandidateChains( std::vector<size_t>& chains, std::unordered_map<size_t, std::set<size_t>>& conflicts, size_t chainsGoal) {
  candidatesGoal = chainsGoal;
  recChainGen({}, chains, conflicts);
}
   
void InitialChainBuilder::findChainConflicts(
    std::vector<size_t> &chains,
    std::unordered_map<size_t, std::vector<size_t>> &nodeConflicts,
    std::unordered_map<size_t, std::set<size_t>> &chainConflicts) {

  for (int i = 0; i < chains.size(); i++) {
    auto idx = chains[i];
    auto& chain = scalarChains[idx]; 
    for (auto& node : chain) {
      auto& conflictsAtNode = nodeConflicts[node];
      chainConflicts[idx].insert(conflictsAtNode.begin(), conflictsAtNode.end());
    }

    for (int i2 = i+1; i2 < chains.size(); i2++) {
      auto idx2 = chains[i2];
      auto &chain2 = scalarChains[idx2];
      if (dependsOn[chain[0]].find(chain2[0]) != dependsOn[chain[0]].end() ||
          dependsOn[chain2[0]].find(chain[0]) != dependsOn[chain2[0]].end()) {
	chainConflicts[idx].insert(idx2);
	chainConflicts[idx2].insert(idx);
      }
    }
  }
}


void
InitialChainBuilder::findConflicts(std::vector<size_t> &chains,
				   std::unordered_map<size_t, std::vector<size_t>> &conflicts) {
  for (auto id : chains) {
    auto& chain = scalarChains[id];
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
  parentChildrenHistoMap[prevNodes->back()].push_back(ownId);
  leafs.push_back(ownId);
  coveredPreLeafNodes.insert(parentId);
}

void InitialChainBuilder::visitGauss(Gauss &n, arg_t arg) {
  size_t ownId = nodes.size()-1;
  auto parentId = childParentMap[ownId];
  auto it = coveredPreLeafNodes.find(parentId);
  if (it != coveredPreLeafNodes.end())
    return;
  std::vector<size_t>* prevNodes = (std::vector<size_t>*) arg.get();
  parentChildrenGaussMap[prevNodes->back()].push_back(ownId);
  leafs.push_back(ownId);
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
