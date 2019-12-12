#include "SchedulingConflictTraversal.h"

std::vector<size_t> SchedulingConflictTraversal::updateConflicts(GraphIRNode& n, arg_t arg) {
  size_t id = conflicts.size();
  auto* path = (std::vector<size_t>*)arg.get();
  conflicts.push_back({*path, {}});
  conflicts[id].first.push_back(id);
  idMap.insert({n.id(), id});
  names.push_back(n.id());
  for (auto& n: *path) {
    conflicts[n].second.push_back(id);
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
  if (pack.size() == simdWidth) {
    size_t id = vecVars.size();
    std::string name;
    for (auto& s : pack) {
      partOf[s].push_back(id);
      name += names[s] + "_";
    }
    vecVars.push_back({pack, model.addVar(0.0, 1.0, 0.0, GRB_BINARY, name)});
    
    return;
  }

  for (auto n : availableOps) {
    std::vector<size_t> newPack(pack);
    newPack.push_back(n);
    std::vector<size_t> newAvailOps;

    std::vector<size_t> exclude(conflicts[n].first.begin(), conflicts[n].first.end());
    exclude.insert(exclude.end(), conflicts[n].second.begin(), conflicts[n].second.end());
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
    serVars[n] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "_" + names[n]);
    std::vector<size_t> newAvailableOps;
    
    std::vector<size_t> exclude(conflicts[n].first.begin(), conflicts[n].first.end());
    exclude.insert(exclude.end(), conflicts[n].second.begin(), conflicts[n].second.end());

    // also cut off all ops < n
    // --> all packs are strictly monotone <=> each combination w/o order is generated only once
    std::set_difference(
        std::lower_bound(availableOps.begin(), availableOps.end(), n),
        availableOps.end(), exclude.begin(), exclude.end(),
        std::inserter(newAvailableOps, newAvailableOps.begin()));
    genVarsRec(pack, newAvailableOps);
  }
}


void SchedulingConflictTraversal::generateVariables() {
  serVars.resize(idMap.size());
  partOf.resize(idMap.size());
  genVarsRoot(sums);
  genVarsRoot(weightedSums);
  genVarsRoot(products);
}
