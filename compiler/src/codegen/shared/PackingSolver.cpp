#include "codegen/shared/PackingSolver.h"
#include "codegen/shared/SchedulingConflictTraversal.h"
#include "gurobi_c++.h"

std::unordered_map<std::string, size_t>
PackingSolver::getPacking(IRGraph &graph, size_t width) {
  GRBEnv env = GRBEnv(true);
  env.set("LogFile", "mip1.log");
  env.start();

  // Create an empty model
  GRBModel model = GRBModel(env);

  SchedulingConflictTraversal sct(width, model);
  sct.findConflicts(graph.rootNode);
  
  sct.generateVariables();

  std::cout << "#vars " << sct.vecVars.size()+sct.serVars.size() << std::endl;
  /*
  for (int i = 0; i < 10; i++) {
    for (auto& l : sct.vecVars[(sct.vecVars.size()/10)*i].lanes) {
      std::cout << " " << sct.names[l] << " ";
    }
    std::cout << std::endl;
  }
  */
  class IndexRefMapper : public BaseVisitor {

  public:
    IndexRefMapper(std::unordered_map<std::string, size_t>& idIndexMap)
        : idMap(idIndexMap) {}
    void buildMap(const NodeReference &rootNode) {
      nodeRefs.resize(idMap.size());
      nodeRefs[idMap[rootNode->id()]] = rootNode;
      rootNode->accept(*this, {});
    }

    void visitHistogram(Histogram &n, arg_t arg) {}

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

  

  IndexRefMapper irm(sct.idMap);
  irm.buildMap(graph.rootNode);

  class InputProducer : public BaseVisitor {

  public:
    InputProducer(std::unordered_map<std::string, size_t> &idIndexMap)
        : idMap(idIndexMap) {}
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
    std::unordered_map<std::string, size_t> &idMap;
    std::vector<std::string> inputs;
  };

  InputProducer ip(sct.idMap);
  GRBQuadExpr obj = 0.0;
  for (int vecIndex = 0; vecIndex < sct.vecVars.size(); vecIndex++) {
    auto &v = sct.vecVars[vecIndex];
    std::unordered_set<size_t> coveredVecInputs;

    std::vector<std::pair<std::vector<size_t>, size_t>> vInputs;

    for (auto &l : v.lanes) {
      irm.nodeRefs[l]->accept(ip, {});
      std::vector<size_t> idVec;
      size_t histogramInputs = 0;
      for (auto &name : ip.inputs) {
	if (sct.idMap.find(name) == sct.idMap.end())
	  histogramInputs++;
	else
	  idVec.push_back(sct.idMap[name]);
      }
      vInputs.push_back({idVec, histogramInputs});
    }

    // Account for cost from histogram inputs
    // Assumption for now: #n wide gather load takes as long as #n loads, but
    // doesn't need insert instr's, so they're equivalent and we don't calculate
    // load costs, only the cost (savings) of vectorizing the arithmetic
    // operations, they're an input for
    size_t vectorizableHistogramInputs = std::numeric_limits<size_t>::max();
    for (auto& l : vInputs) {
      vectorizableHistogramInputs = std::min(vectorizableHistogramInputs, l.second);
    }

    if (vectorizableHistogramInputs >= 1) {
      // To combine #vectorizableHistogramInputs values we need
      // #vectorizableHistogramInputs-1 arithmetic operations
      obj += v.var*(vectorizableHistogramInputs-1);
    }

    // all non vectorizable inputs must perform their arithmetic operation scalar
    for (auto& l : vInputs) {
      size_t scalarHistInputs = l.second-vectorizableHistogramInputs;
      obj += v.var*(scalarHistInputs);
      // Note: the above overapproximates the # of necessary arithmetic
      // operations by 1, which cancels out with the necessary insertion of
      // the result scalar into the vector v
    }

    
    // we assume one arithmetic operation per input, however, for the two
    // inputs, only one arithmetic operation is needed, thus we substract one
    // TODO this isn't really accurate, e.g. in cases where all inputs are scalar
    // Better: only substract if at least (?) two covering vec inputs are selected
    obj -= v.var*1;

    // TODO
    // Most significant inaccuracies at the moment:
    // Multiple values (one for each lane) can be inserted with one instr
    // Multiple scalar values for a single lane can be combinded beforehand and
    // thus need only one insertion
    for (int i = 0; i < v.lanes.size(); i++) {
      std::vector<size_t> possibleInputVecs;
      for (auto &inputID : vInputs[i].first) {
        auto &inVecs = sct.partOf[inputID];
        possibleInputVecs.insert(possibleInputVecs.end(), inVecs.begin(),
                                 inVecs.end());
      }

      for (auto &vec : possibleInputVecs) {
        if (i != 0) {
          if (coveredVecInputs.find(vec) != coveredVecInputs.end()) {
            // we already handled vec when looking at lane 0 and found it to be
            // directly compatible
            // Thus we do not need to account for cost of the combo v x vec
            // again
            continue;
          }
        } else {
          // check if vec can be direct input to v
          auto &valuesProvidedBy_vec_ = sct.vecVars[vec].lanes;
          bool foundInputsForAllLanes = true;
          for (int j = 0; j < v.lanes.size(); j++) {
            auto &neededInputsForThisLane = vInputs[j].first;
            bool foundInputForLane = false;

            // TODO: optimize, e.g. build intersection of all #width partOfs
            for (auto &ni : neededInputsForThisLane) {
              for (auto &pv : valuesProvidedBy_vec_) {
                if (ni == pv) {
                  foundInputForLane = true;
                  break;
                }
              }
              if (foundInputForLane) {
                break;
              }
            }
            if (!foundInputForLane) {
              foundInputsForAllLanes = false;
            }
          }

          // TODO: it's not always necessary for all width values to be
          // needed to be cheaper than the default case (extract all)
          // Example:
          // Width 4, 3 values are needed, one extract + one shuffle (?) + one
          // 0 or 1 insert (neutral element) is still cheaper than extracting
          // all 4 vals
          if (foundInputsForAllLanes) {
            // vec does not need to be extracted, it can be directly
            // multiplied onto v
            coveredVecInputs.insert(vec);
            // assume cost of one for this mult if both are selected
            obj += sct.vecVars[vec].var * v.var * 1;
              continue;
          }
        }
        // if we get here, we will need to extract the value from vec, perform
        // the arith and insert it into the starting vector for v
        // -> cost estimate: 3
        obj += sct.vecVars[vec].var * v.var * 3;
      }

      for (auto& inputVal : vInputs[i].first) {
	// Only arith and insert necessary
	// -> cost estimate: 2
	obj += v.var*sct.serVars[inputVal]*2;
      }
    }

    directVecInputs.insert({vecIndex, coveredVecInputs});
  }

  for (int i = 0; i < sct.serVars.size(); i++) {
    irm.nodeRefs[i]->accept(ip, {});
    std::vector<size_t> idVec;
    size_t histogramInputs = 0;
    for (auto &name : ip.inputs) {
      if (sct.idMap.find(name) == sct.idMap.end())
        histogramInputs++;
      else
        idVec.push_back(sct.idMap[name]);
    }
    // we assume one arithmetic operation per input, however, for the two
    // inputs, only one arithmetic operation is needed, thus we substract one
    obj -= sct.serVars[i]*1;

    // Each histogram input takes one arithmetic instruction
    obj += sct.serVars[i]*histogramInputs;
    
    for (auto& input : idVec) {
      // when both values are not in a vector, there is no extra cost besides
      // the arithmetic operation
      obj += sct.serVars[i]*sct.serVars[input]*1;
      for (auto& vecIn : sct.partOf[input]) {
        // cost: extract + arith = 2
	obj += sct.serVars[i]*sct.vecVars[vecIn].var*2;
      }
    }
  }

  model.setObjective(obj, GRB_MINIMIZE);

  // for (x,y) e vecVars x vecVars
  // if exists l1 in x that is needed in y and l2 in y that is needed in x,
  // x+y<=1
  for (auto &vec : sct.vecVars) {
    for (auto &l : vec.lanes) {
      for (auto &dep : sct.conflicts[l].second) {
        auto &possiblyConflictingVecs = sct.partOf[dep];
        for (auto &depVec : possiblyConflictingVecs) {
          bool conflict = false;
          for (auto &depVecLane : sct.vecVars[depVec].lanes) {
            for (auto &reverseDep : sct.conflicts[depVecLane].second) {
              for (auto l2 : vec.lanes) {
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
          model.addConstr(sct.vecVars[dep].var + sct.vecVars[depVec].var <= 1);
        }
      }
    }
  }

  // make sure each value is calculated only once
  for (int i = 0; i < sct.serVars.size(); i++) {
    GRBLinExpr opSum = 0.0;
    opSum += sct.serVars[i];
    for (auto& vec : sct.partOf[i]) {
      opSum += sct.vecVars[vec].var;
    }
    model.addConstr(opSum == 1);
  }

  model.optimize();
  std::unordered_map<std::string, size_t> res;

  for (int i = 0; i < sct.vecVars.size(); i++) {
    if (sct.vecVars[i].var.get(GRB_DoubleAttr_X) > 0.1) {
      std::vector<NodeReference> laneRefs;
      for (auto& l : sct.vecVars[i].lanes) {
	laneRefs.push_back(irm.nodeRefs[l]);
	res.insert({sct.names[l], i});
      }
      vectors.insert({i, laneRefs});
      std::cout << "selected ";
      for (auto& l : sct.vecVars[i].lanes) {
	std::cout << sct.names[l] << ", ";
      }
      std::cout << std::endl;
    }
  }

  for (int i = 0; i < sct.serVars.size(); i++) {
    if (sct.serVars[i].get(GRB_DoubleAttr_X) > 0.1) {
      std::cout << "selected " << sct.names[i] << std::endl;
    }
  }

  return res;
}
