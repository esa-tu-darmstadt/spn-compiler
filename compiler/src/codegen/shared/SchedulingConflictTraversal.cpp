#include "SchedulingConflictTraversal.h"

std::vector<size_t>
flattenPack(std::vector<size_t> base, size_t limit,
            std::unordered_map<size_t, std::vector<size_t>> &fixedPacks) {
  std::queue<size_t> components;
  for (auto& e : base) {
    components.push(e);
  }
  std::vector<size_t> res;

  while (!components.empty()) {
    auto &t = components.front();
    if (t < limit) {
      res.push_back(t);
    } else {
      for (auto &comp2 : fixedPacks[t]) {
        components.push(comp2);
      }
    }
    components.pop();
  }
  return res;
}
std::vector<size_t> SchedulingConflictTraversal::updateConflicts(GraphIRNode& n, arg_t arg) {
  size_t id = conflicts.size();
  serVars.insert({id, model->addVar(0.0, 1.0, 0.0, GRB_BINARY)});
  auto* path = (std::vector<size_t>*)arg.get();
  conflicts.push_back({{path->begin(), path->end()}, {}});
  conflicts[id].first.insert(id);
  idMap.insert({n.id(), id});
  names.push_back(n.id());
  for (auto& n: *path) {
    conflicts[n].second.insert(id);
  }
  path->push_back(id);
  return *path;
}

void SchedulingConflictTraversal::visitInputvar(InputVar &n, arg_t arg) {}
void SchedulingConflictTraversal::visitHistogram(Histogram &n, arg_t arg) {}
void SchedulingConflictTraversal::visitProduct(Product &n, arg_t arg) {
  auto p = updateConflicts(n, arg);
  products.push_back(p[p.size()-1]);
  for (auto &c : *n.multiplicands()) {
    auto new_path = std::make_shared<std::vector<size_t>>(p);
    c->accept(*this, new_path);
  }
}

void SchedulingConflictTraversal::visitSum(Sum &n, arg_t arg) { 
  auto p = updateConflicts(n, arg);
  sums.push_back(p[p.size()-1]);
  for (auto &c : *n.addends()) {
    auto new_path = std::make_shared<std::vector<size_t>>(p);
    c->accept(*this, new_path);
  }
}

void SchedulingConflictTraversal::visitWeightedSum(WeightedSum &n, arg_t arg) {
  auto p = updateConflicts(n, arg);
  weightedSums.push_back(p[p.size()-1]);
  for (auto &c : *n.addends()) {
    auto new_path = std::make_shared<std::vector<size_t>>(p);
    c.addend->accept(*this, new_path);
  }
}

void SchedulingConflictTraversal::findConflicts(NodeReference root) {
  // maybe use a list here
  auto path = std::make_shared<std::vector<size_t>>();
  root->accept(*this, path);
}

void SchedulingConflictTraversal::genVarsRec(std::vector<size_t> pack, std::vector<size_t> availableOps) {
  if (pack.size() > 1) {
    size_t id = vecVars.size();
    
    std::string name;
    for (auto& s : flattenPack(pack, names.size(), fixedPacks)) {
      partOf[s].push_back(id);
      name += names[s] + "_";
    }
    vecVars.push_back({pack, model->addVar(0.0, 1.0, 0.0, GRB_BINARY, name)});
    if (pack.size() == simdWidth)
      return;
  }

  for (auto n : availableOps) {
    std::vector<size_t> newPack(pack);
    newPack.push_back(n);
    std::vector<size_t> newAvailOps;

    std::set<size_t> exclude(conflicts[n].first.begin(), conflicts[n].first.end());
    exclude.insert(conflicts[n].second.begin(), conflicts[n].second.end());
    // TODO assert that conflicts and availableOps is sorted
    std::set_difference(
        std::lower_bound(availableOps.begin(), availableOps.end(), n),
        availableOps.end(), exclude.begin(), exclude.end(),
        std::inserter(newAvailOps, newAvailOps.begin()));

    genVarsRec(newPack, newAvailOps);
  }
}


void SchedulingConflictTraversal::genVarsRoot(std::vector<size_t> availableOps) {
  for (auto n : availableOps) {
    std::vector<size_t> pack;
    pack.push_back(n);
    std::vector<size_t> newAvailableOps;
    
    std::set<size_t> exclude(conflicts[n].first.begin(), conflicts[n].first.end());
    exclude.insert(conflicts[n].second.begin(), conflicts[n].second.end());

    // also cut off all ops < n
    // --> all packs are strictly monotone <=> each combination w/o order is generated only once
    std::set_difference(
        std::lower_bound(availableOps.begin(), availableOps.end(), n),
        availableOps.end(), exclude.begin(), exclude.end(),
        std::inserter(newAvailableOps, newAvailableOps.begin()));
    genVarsRec(pack, newAvailableOps);
  }
}

std::vector<size_t> SchedulingConflictTraversal::getNewOpSet(
    std::vector<size_t> &old,
    std::unordered_map<size_t, size_t> &componentToPackId) {
  // collect them independently so that in the end, newOpsPack is appended to
  // newOps and newOps is still ordered
  std::vector<size_t> newOps;
  std::set<size_t> newOpPacks;
  for (auto& opNode : old) {
    auto it = componentToPackId.find(opNode);
    if (it == componentToPackId.end()) {
      // opNode was not packed with another op node in the previous iteration
      newOps.push_back(opNode);
      for (auto& userConflict : conflicts[opNode].first) {
	if (componentToPackId.find(userConflict) != componentToPackId.end()) {
	  conflicts[opNode].first.insert(componentToPackId[userConflict]);
	}
      }
      for (auto& inputConflict : conflicts[opNode].second) {
	if (componentToPackId.find(inputConflict) != componentToPackId.end()) {
          // Note Do not remove "old" _inputConflict_, since the constraint
          // builder for mutually exclusive vector variables relies on single
          // values being present in _conflicts[opNode].second_
          conflicts[opNode].second.insert(componentToPackId[inputConflict]);
	}
      }
    } else if (newOpPacks.find(it->second) == newOpPacks.end()) {
      auto& confs = conflicts[it->second];
      // newId cannot be packed with any of the packs or singles that any of
      // its components had conflicts with
      std::set<size_t> conflictUses;
      conflictUses.insert(it->second);
      std::set<size_t> conflictInputs;
      for (auto component : fixedPacks[it->second]) {
	auto& vecPair = conflicts[component];
	// insert conflicts either from old or new id, if they're now packaged
	for (auto& use : vecPair.first) {
	  if (componentToPackId.find(use) != componentToPackId.end()) {
	    conflictUses.insert(componentToPackId[use]);
	  } else {
	    conflictUses.insert(use);
	  }
	  
	}
	for (auto& input : vecPair.second) {
	  if (componentToPackId.find(input) != componentToPackId.end()) {
	    conflictInputs.insert(componentToPackId[input]);
	  } else {
	    conflictInputs.insert(input);
	  }
	  
	}
      }
      confs.first = conflictUses;
      confs.second = conflictInputs;
      newOpPacks.insert(it->second);
    }
  }
  newOps.insert(newOps.end(), newOpPacks.begin(), newOpPacks.end());
  return newOps;
}

void SchedulingConflictTraversal::setupNewIteration(
    std::unordered_set<size_t> prevVecs, std::vector<size_t> preNonVecs,
    GRBModel* newModel) {
  model = newModel;
  partOf.clear();
  std::unordered_map<size_t, size_t> componentToPackId;
  std::vector<vecVar> newVecVars;
  std::unordered_map<size_t, size_t> newOldVecs;
  // TODO merge serVars and vecVars (serVars are vecVars with single element
  // lane vectors)

  for (auto& prevVec : prevVecs) {
    auto it = oldVecs.find(prevVec);
    size_t fixedPacksIdx;
    if (it != oldVecs.end()) {
      // prevVec was already a vector before the last iteration, thus we do
      // not need to add it to conflicts
      fixedPacksIdx = it->second;
    } else {
      size_t newId = conflicts.size();
      for (auto &component : vecVars[prevVec].lanes) {
        componentToPackId.insert({component, newId});
      }
      fixedPacks.insert({newId, vecVars[prevVec].lanes});
      conflicts.push_back({});
      fixedPacksIdx = newId;
    }
    size_t vecVarId = newVecVars.size();
    newOldVecs.insert({vecVarId, fixedPacksIdx});
    std::string name;
    for (auto& lane : flattenPack(vecVars[prevVec].lanes, names.size(), fixedPacks)) {
      name += names[lane];
      partOf[lane].push_back(vecVarId);
      singleOpToFixedVec.insert({lane, fixedPacksIdx});
    }
    newVecVars.push_back({vecVars[prevVec].lanes, model->addVar(0.0, 1.0, 0.0, GRB_BINARY)});
  }
  vecVars = newVecVars;
  oldVecs= newOldVecs;

  
  std::unordered_map<size_t, GRBVar> newSerVars;
  for (auto& nonVec : preNonVecs) {
    newSerVars.insert({nonVec, model->addVar(0.0, 1.0, 0.0, GRB_BINARY)});
  }

  serVars = newSerVars;
  sums = getNewOpSet(sums, componentToPackId);
  weightedSums = getNewOpSet(weightedSums, componentToPackId);
  products = getNewOpSet(products, componentToPackId);
}

void SchedulingConflictTraversal::generateVariables() {
  genVarsRoot(sums);
  genVarsRoot(weightedSums);
  genVarsRoot(products);
}
