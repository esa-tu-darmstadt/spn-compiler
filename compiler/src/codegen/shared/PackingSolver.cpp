#include "codegen/shared/PackingSolver.h"
#include "gurobi_c++.h"

extern llvm::cl::OptionCategory SPNCompiler;

llvm::cl::opt<bool> incSolve(
    "incSolve",
    llvm::cl::desc("Use faster ILP process that builds vectors incrementally"),
    llvm::cl::cat(SPNCompiler));

#define CONSTR_EXCLUSIONS false

class IndexRefMapper : public BaseVisitor {

public:
  IndexRefMapper(std::unordered_map<std::string, size_t> &idIndexMap)
      : idMap(idIndexMap) {}
  void buildMap(const NodeReference &rootNode) {
    nodeRefs.resize(idMap.size());
    nodeRefs[idMap[rootNode->id()]] = rootNode;
    rootNode->accept(*this, {});
  }

  void visitHistogram(Histogram &n, arg_t arg) {}
  void visitGauss(Gauss &n, arg_t arg) {}

  void visitProduct(Product &n, arg_t arg) {
    for (auto &c : *n.multiplicands()) {
      if (idMap.find(c->id()) == idMap.end())
        continue;
      nodeRefs[idMap[c->id()]] = c;
      c->accept(*this, {});
    }
  }

  void visitSum(Sum &n, arg_t arg) {
    for (auto &c : *n.addends()) {
      if (idMap.find(c->id()) == idMap.end())
        continue;
      nodeRefs[idMap[c->id()]] = c;
      c->accept(*this, {});
    }
  }

  void visitWeightedSum(WeightedSum &n, arg_t arg) {
    for (auto &c : *n.addends()) {
      if (idMap.find(c.addend->id()) == idMap.end())
        continue;
      nodeRefs[idMap[c.addend->id()]] = c.addend;
      c.addend->accept(*this, {});
    }
  }
  std::unordered_map<std::string, size_t> &idMap;
  std::vector<NodeReference> nodeRefs;
};

solverResult PackingSolver::runSolver(
       std::vector<std::pair<std::set<size_t>, std::set<size_t>>> &conflicts,
       std::unordered_map<std::string, size_t> &idMap,
       std::vector<vecVar> &vecVars,
       std::unordered_map<size_t, std::vector<size_t>> &partOf,
       std::unordered_map<size_t, GRBVar> &serVars,
       std::unordered_map<size_t, std::vector<size_t>> &fixedPacks,
       std::unordered_set<std::string>& histograms, std::unordered_set<std::string>& gaussians,
       IndexRefMapper &irm,
       std::unordered_map<size_t, size_t> &singleOpToFixedVec, GRBModel& model) {

  class InputProducer : public BaseVisitor {

  public:
    void visitProduct(Product &n, arg_t arg) {
      inputs.clear();
      for (auto &c : *n.multiplicands()) {
	inputs.push_back(c->id());
      }
    }

    void visitSum(Sum &n, arg_t arg) {
      inputs.clear();
      for (auto &c : *n.addends()) {
	inputs.push_back(c->id());
      }
    }

    void visitWeightedSum(WeightedSum &n, arg_t arg) {
      inputs.clear();
      for (auto &c : *n.addends()) {
	inputs.push_back(c.addend->id());
      }
    }
    std::vector<std::string> inputs;
  };

  InputProducer ip;
  GRBQuadExpr obj = 0.0;
  // if first is activated, only one of second can be activated
  std::unordered_map<size_t, std::vector<size_t>> exclusions;

  std::unordered_map<size_t, std::unordered_set<size_t>> directVecInputMap;
  for (int vecIndex = 0; vecIndex < vecVars.size(); vecIndex++) {
    auto &v = vecVars[vecIndex];
    std::unordered_set<size_t> coveredVecInputs;

    std::vector<std::vector<size_t>> vInputs;
    std::multiset<size_t, std::greater<size_t>> inputHistos;
    std::multiset<size_t> inputGaussians;
    auto flatLanes = flattenPack(v.lanes, irm.nodeRefs.size(), fixedPacks);
    size_t vecWidth = flatLanes.size();

    for (auto &l : flatLanes) {
      irm.nodeRefs[l]->accept(ip, {});
      std::vector<size_t> idVec;
      size_t histogramInputs = 0;
      size_t gaussInputs = 0;
      for (auto &name : ip.inputs) {
	if (histograms.find(name) != histograms.end())
	  histogramInputs++;
	else if (gaussians.find(name) != gaussians.end())
	  gaussInputs++;
	else
	  idVec.push_back(idMap[name]);
      }
      vInputs.push_back(idVec);
      inputHistos.insert(histogramInputs);
      inputGaussians.insert(gaussInputs);
    }
    
    obj+= v.var*ci->histogramCost(inputHistos);

    obj+= v.var*ci->gaussCost(inputGaussians);
    
    
    // we assume one arithmetic operation per input, however, for the first two
    // inputs, only one arithmetic operation is needed, thus we substract one
    // TODO this isn't really accurate, e.g. in cases where all inputs are scalar
    // Better: only substract if at least (?) two covering vec inputs are selected
    obj -= v.var*ci->vecArithCost;

    // TODO
    // Most significant inaccuracies at the moment:
    // Multiple values (one for each lane) can be inserted with one instr (?)
    // Multiple scalar values for a single lane can be combinded beforehand and
    // thus need only one insertion
    std::vector<std::vector<size_t>> possibleInputVecsPerLane;
    std::unordered_set<size_t> allInputVecs;
    for (int i = 0; i < vInputs.size(); i++) {
      std::vector<size_t> possibleInputVecs;
      for (auto &inputID : vInputs[i]) {
        auto &inVecs = partOf[inputID];
        possibleInputVecs.insert(possibleInputVecs.end(), inVecs.begin(),
                                 inVecs.end());
      }
      possibleInputVecsPerLane.push_back(possibleInputVecs);
      allInputVecs.insert(possibleInputVecs.begin(), possibleInputVecs.end());
    }

    for (auto& vec : allInputVecs) {
      // check if vec can be direct input to v
      auto valuesProvidedBy_vec_ =
          flattenPack(vecVars[vec].lanes, irm.nodeRefs.size(), fixedPacks);
      int foundInputsForLanes = 0;
      std::vector<size_t> inputIds;
      for (int j = 0; j < vInputs.size(); j++) {
        auto &neededInputsForThisLane = vInputs[j];
        bool foundInputForLane = false;

        // TODO: optimize, e.g. build intersection of all #width partOfs
        for (auto &ni : neededInputsForThisLane) {
          for (auto &pv : valuesProvidedBy_vec_) {
            if (ni == pv) {
              foundInputForLane = true;
              inputIds.push_back(pv);
              break;
            }
          }
          if (foundInputForLane) {
            break;
          }
        }
        if (foundInputForLane) {
          foundInputsForLanes++;
        }
      }

      if (foundInputsForLanes == vInputs.size()) {
        // vec does not need to be extracted, it can be directly
        // multiplied onto v as all values provided by vec are needed by v
        coveredVecInputs.insert(vec);
        directVecInputMap[vecIndex].insert(vec);
        exclusions[vec].push_back(vecIndex);
        // assume cost of one for this op if both are selected
        obj += vecVars[vec].var * v.var * ci->vecArithCost;
      } else if (foundInputsForLanes *
                     (ci->getExtractCost(valuesProvidedBy_vec_.size()) +
                      ci->scalarArithCost + ci->getInsertCost(vInputs.size())) >
                 (vInputs.size() - foundInputsForLanes) *
                         ci->getInsertCost(valuesProvidedBy_vec_.size()) +
                     ci->vecArithCost) {
        // In the general case, we would need _foundInputsForLanes_ * extract,
        // arith, insert  operations if vec and v are selected
        // An alternative, which is is cheaper if this branch is taken, is
        // to extract the not needed values beforehand (this cost will be
        // accounted for by the nodes that need these values), insert
        // neutral values into those lanes and then use the vector as a whole
        obj += vecVars[vec].var * v.var *
               ((vInputs.size() - foundInputsForLanes) *
                    ci->getInsertCost(valuesProvidedBy_vec_.size()) +
                ci->vecArithCost);
        coveredVecInputs.insert(vec);
        directVecInputMap[vecIndex].insert(vec);
        // TODO For now we will allow one user of vec to use this shortcut,
        // although with copies, half extracts, register renaming etc. this
        // could be modelled more precisely
        exclusions[vec].push_back(vecIndex);
      }
    }

    for (int i = 0; i < vInputs.size(); i++) {
      auto &possibleInputVecs = possibleInputVecsPerLane[i];
      for (auto &vec : possibleInputVecs) {
        auto valuesProvidedBy_vec_ =
            flattenPack(vecVars[vec].lanes, irm.nodeRefs.size(), fixedPacks);
        if (coveredVecInputs.find(vec) != coveredVecInputs.end()) {
          // we already handled vec when looking at lane 0 and found it to be
          // directly compatible
          // Thus we do not need to account for cost of the combo v x vec
          // again
          continue;
        }
        // if we get here, we will need to extract the value from vec, perform
        // the arith and insert it into the starting vector for v
        obj += vecVars[vec].var * v.var *
               (ci->getExtractCost(valuesProvidedBy_vec_.size()) +
                ci->scalarArithCost + ci->getInsertCost(vInputs.size()));
      }

      for (auto& inputVal : vInputs[i]) {
	auto singleOpMapIt = singleOpToFixedVec.find(inputVal);
	if (singleOpMapIt == singleOpToFixedVec.end()) {
	  // inputVal is still a single value, thus only arith and insert necessary
          // -> cost estimate: 2
          obj += v.var * serVars[inputVal] *
                 (ci->scalarArithCost + ci->getInsertCost(vInputs.size()));
        }
      }
    }
  }

  for (auto &serVar : serVars) {
    // serVar represents a single value
    irm.nodeRefs[serVar.first]->accept(ip, {});
    std::vector<size_t> idVec;
    size_t histogramInputs = 0;
    size_t gaussInputs = 0;
    for (auto &name : ip.inputs) {
      if (histograms.find(name) != histograms.end())
        histogramInputs++;
      else if (gaussians.find(name) != gaussians.end())
        gaussInputs++;
      else
        idVec.push_back(idMap[name]);
    }
    // we assume one arithmetic operation per input, however, for the two
    // inputs, only one arithmetic operation is needed, thus we substract one
    obj -= serVar.second * ci->scalarArithCost;

    // Each histogram input takes one arithmetic instruction
    obj += serVar.second * histogramInputs * ci->scalarArithCost;
    
    obj += serVar.second * gaussInputs * (ci->gaussArithCost + ci->scalarArithCost);

    for (auto &input : idVec) {
      auto singleOpMapIt = singleOpToFixedVec.find(input);
      if (singleOpMapIt == singleOpToFixedVec.end()) {
        // when both values are not in a vector, there is no extra cost
        // besides the arithmetic operation
        obj += serVar.second * serVars[input] * ci->scalarArithCost;
      }
      for (auto &vecIn : partOf[input]) {
        // cost: extract + arith = 2
        obj += serVar.second * vecVars[vecIn].var *
               (ci->getExtractCost(flattenPack(vecVars[vecIn].lanes,
                                               irm.nodeRefs.size(), fixedPacks)
                                       .size()) +
                ci->scalarArithCost);
      }
    }
  }

  model.setObjective(obj, GRB_MINIMIZE);

  // for (x,y) e vecVars x vecVars
  // if exists l1 in x that is needed in y and l2 in y that is needed in x,
  // x+y<=1
  for (auto &vec : vecVars) {
    for (auto &l : flattenPack(vec.lanes, irm.nodeRefs.size(), fixedPacks)) {
      for (auto &dep : conflicts[l].second) {
        // Note: partOf[dep] may return an empty vector if dep represents a
        // fixed vec but since the components of dep are also in
        // _conflicts[l].second_, this is no problem
        auto &possiblyConflictingVecs = partOf[dep];
        for (auto &depVec : possiblyConflictingVecs) {
          bool conflict = false;
          for (auto &depVecLane : flattenPack(vecVars[depVec].lanes, irm.nodeRefs.size(), fixedPacks)) {
            for (auto &reverseDep : conflicts[depVecLane].second) {
              for (auto l2 : flattenPack(vec.lanes, irm.nodeRefs.size(), fixedPacks)) {
                if (l2 == reverseDep) {
                  conflict = true;
                  break;
                }
              }
              if (conflict)
                break;
            }

            if (conflict)
              break;
          }
          if (!conflict)
            continue;

          // vec needs a value from depVec and depVec needs a value from vec,
          // thus they cannot both be selected at the same time
          model.addConstr(vecVars[dep].var + vecVars[depVec].var <= 1);
        }
      }
    }
  }

  // make sure each value is calculated only once
  for (int i = 0; i < irm.nodeRefs.size(); i++) {
    GRBLinExpr opSum = 0.0;
    if (singleOpToFixedVec.find(i) == singleOpToFixedVec.end())
      opSum += serVars[i];
    
    for (auto& vec : partOf[i]) {
      opSum += vecVars[vec].var;
    }
    model.addConstr(opSum == 1);
  }
  if (CONSTR_EXCLUSIONS) {
    for (auto &ex : exclusions) {
      GRBQuadExpr opSum = 0.0;
      for (auto &vec : ex.second) {
        opSum += vecVars[ex.first].var * vecVars[vec].var;
      }
      model.addQConstr(opSum <= 1);
    }
  }
  model.optimize();
  model.write("test.lp");
  std::unordered_set<size_t> vecs;
  std::vector<size_t> nonVecs;

  for (int i = 0; i < vecVars.size(); i++) {
    if (vecVars[i].var.get(GRB_DoubleAttr_X) > 0.1) {
      vecs.insert(i);
    }
  }
  for (auto &serVar : serVars) {
    if (serVar.second.get(GRB_DoubleAttr_X) > 0.1) {
      nonVecs.push_back(serVar.first);
    }
  }
  std::unordered_map<size_t,std::unordered_set<size_t>> directVecs;
  std::unordered_set<size_t> usedDirectVecInputs;
  for (auto& vec : vecs) {
    for (auto& direct : directVecInputMap[vec]) {
      if (vecs.find(direct) != vecs.end() 
	  // each vector can only be direct input to one vector
	  // TODO reflect this in the cost model
	  ) {
        // both _vec_ and _direct.first_ were selected, so we store _direct_ as a direct input
        if (usedDirectVecInputs.find(direct) !=
            usedDirectVecInputs.end()) {
	  std::cout << "double use" << std::endl;
        } else {
          directVecs[vec].insert(direct);
          usedDirectVecInputs.insert(direct);
        }
      }
    }
  }
  
  return {vecs,nonVecs, directVecs};
}

vectorizationResultInfo
PackingSolver::getVectorization(IRGraph &graph, size_t width) {
  GRBEnv env = GRBEnv(true);
  env.set("LogFile", "mip1.log");
  env.start();

  // Create an empty model
  GRBModel model = GRBModel(env);

  int initWidth;

  if (incSolve)
    initWidth = 2;
  else
    initWidth = width;

  SchedulingConflictTraversal sct(initWidth, &model);
  sct.findConflicts(graph.rootNode);
  
  sct.generateVariables();

  model.update();
  
  std::cout << "#vars " << sct.vecVars.size()+sct.serVars.size() << std::endl;

  IndexRefMapper irm(sct.idMap);
  irm.buildMap(graph.rootNode);
  ci = std::make_unique<CostInfo>(initWidth);

  auto packing = runSolver(sct.conflicts, sct.idMap, sct.vecVars, sct.partOf,
                           sct.serVars, sct.fixedPacks, sct.histograms,
                           sct.gaussians, irm, sct.singleOpToFixedVec, model);

  std::cout << "selected " << std::endl;
  for (auto i : packing.vecs) {
    for (auto &t : flattenPack(sct.vecVars[i].lanes, irm.nodeRefs.size(),
                               sct.fixedPacks)) {
      std::cout << sct.names[t] << ", ";
    }
    std::cout << std::endl;
  }

  
  for (auto &serVar : packing.nonVecs) {
      std::cout << sct.names[serVar] << std::endl;
  }

  if (incSolve) {
    while (initWidth < width) {
      GRBModel newModel = GRBModel(env);
      sct.setupNewIteration(packing.vecs, packing.nonVecs, &newModel);
      sct.generateVariables();
      newModel.update();
      std::cout << "#vars " << sct.vecVars.size() + sct.serVars.size()
                << std::endl;
      ci = std::make_unique<CostInfo>(initWidth);
      packing = runSolver(sct.conflicts, sct.idMap, sct.vecVars, sct.partOf,
                          sct.serVars, sct.fixedPacks, sct.histograms,
                          sct.gaussians, irm, sct.singleOpToFixedVec, newModel);

      initWidth *=2;

      std::cout << "selected " << std::endl;
      for (auto i : packing.vecs) {
        for (auto &t : flattenPack(sct.vecVars[i].lanes, irm.nodeRefs.size(),
                                   sct.fixedPacks)) {
          std::cout << sct.names[t] << ", ";
        }
        std::cout << std::endl;
      }

      for (auto &serVar : packing.nonVecs) {
        std::cout << sct.names[serVar] << std::endl;
      }
    }
  }

  std::unordered_map<std::string, size_t> res;
  std::unordered_map<size_t, size_t> vecVarIdxToVectorsIdx;

  for (auto &vec : packing.vecs) {
    size_t vectorId = vectors.size();

    vecVarIdxToVectorsIdx.insert({vec, vectorId});

    std::vector<NodeReference> laneRefs;

    for (auto &t : flattenPack(sct.vecVars[vec].lanes, irm.nodeRefs.size(),
                               sct.fixedPacks)) {
      laneRefs.push_back(irm.nodeRefs[t]);
      res.insert({sct.names[t], vectorId});
    }
    vectors.push_back(laneRefs);
  }

  for (auto& vec : packing.directVecInputs) {
    size_t vecIdx = vecVarIdxToVectorsIdx[vec.first];
    for (auto& in : vec.second) {
      size_t inVecIdx = vecVarIdxToVectorsIdx[in];
      directVecInputs[vecIdx].insert(inVecIdx);
    }
  }

  return {res, directVecInputs, vectors};
}
