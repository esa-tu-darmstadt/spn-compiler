#include <unordered_set>
#include <graph-ir/GraphIRNode.h>
#include "llvm/IR/IRBuilder.h"
#include "transform/BaseVisitor.h"
#include "util/GraphIRTools.h"
#include <unordered_map>
#include <set>
#include <iostream>

using namespace llvm;

extern llvm::cl::OptionCategory SPNCompiler;

extern llvm::cl::opt<bool> useGather;
extern llvm::cl::opt<bool> selectBinary;

struct irVal {
  Value* val;
  int pos;
  size_t vec;
};

class IREmitter : public BaseVisitor {
public:
  IREmitter(
      std::unordered_map<std::string, size_t> &vec,
      std::unordered_map<size_t, std::unordered_set<size_t>>& directVecInputs,
      std::vector<std::vector<NodeReference>>& vectors, Value *in,
      Function *func, LLVMContext &context, IRBuilder<> &builder,
      Module *module, unsigned width);

  void visitInputvar(InputVar &n, arg_t arg) override;

  void visitHistogram(Histogram &n, arg_t arg) override;
  void visitGauss(Gauss &n, arg_t arg) override;

  void visitProduct(Product &n, arg_t arg) override;
  void visitSum(Sum &n, arg_t arg) override;
  void visitWeightedSum(WeightedSum &n, arg_t arg) override;

  std::unordered_map<std::string, irVal> getNodeMap();

private:
  Value* getHistogramPtr(Histogram& n);
  template <class T> Value* getNeutralValue();
  template <> Value* getNeutralValue<WeightedSum>() {return constantZero;}
  template <> Value* getNeutralValue<Sum>() {return constantZero;}
  template <> Value* getNeutralValue<Product>() {return constantOne;}

  Value *scalarArith(WeightedSum &n, Value *acc, Value *in, std::string id) {
    double weight = 0.0;
    for (auto& addend : *n.addends()) {
      if (addend.addend->id() == id) {
	weight = addend.weight;
	break;
      }
    }
    assert(weight > 0.0);
    auto mul = _builder.CreateFMul(
        ConstantFP::get(Type::getDoubleTy(_context), weight), in);
    if (acc)
      return _builder.CreateFAdd(acc, mul);
    return mul;
  }
  Value *scalarArith(Sum &n, Value *acc, Value *in, std::string id) {
    if (acc)
      return _builder.CreateFAdd(acc, in);
    return in;
  }
  
  Value *scalarArith(Product &n, Value *acc, Value *in, std::string id) {
    if (acc)
      return _builder.CreateFMul(acc, in);
    return in;
  }

  template <class T>
  Value *handleScalarInput(T &n, Value *acc, NodeReference in) {
    assert(node2value.find(in->id()) != node2value.end());
    auto inInfo = node2value[in->id()];
    Value *inVal;
    if (inInfo.pos == -1)
      inVal = inInfo.val;
    else
      inVal = _builder.CreateExtractElement(inInfo.val, inInfo.pos, in->id() + "_extr");
    return scalarArith(n, acc, inVal, in->id());
  }

  Value *vecArith(WeightedSum &n, Value *acc, Value *in,
                  std::unordered_map<size_t, NodeReference> accOrder,
                  std::vector<std::pair<std::string, size_t>> inOrder) {

    size_t vectorWidth = 0;
    for (auto &l : accOrder)
      vectorWidth = std::max(l.first + 1, vectorWidth);
    
    Value *weights = UndefValue::get(
        VectorType::get(Type::getDoubleTy(_context), vectorWidth));

    for (auto &lane : accOrder) {
      auto inputs = ((WeightedSum *)lane.second.get())->addends();
      double weight = 0.0;
      for (auto &inVal : inOrder) {
        if (inVal.second == lane.first) {
	  for (auto& in : *inputs) {
            if (inVal.first == in.addend->id()) {
	      weight = in.weight;
	    }
          }
          assert(weight > 0.0);
        }
      }
      weights = _builder.CreateInsertElement(weights, ConstantFP::get(Type::getDoubleTy(_context), weight), lane.first);
    }
    auto mul = _builder.CreateFMul(in, weights);
    if (acc)
      return _builder.CreateFAdd(acc, mul);
    return mul;
  }

  Value *vecArith(Sum &n, Value *acc, Value* in,
                  std::unordered_map<size_t, NodeReference> accOrder,
                  std::vector<std::pair<std::string, size_t>> inOrder) {
    if (acc)
      return _builder.CreateFAdd(acc, in);
    return in;
  }

  Value *vecArith(Product &n, Value *acc, Value* in,
                  std::unordered_map<size_t, NodeReference> accOrder,
                  std::vector<std::pair<std::string, size_t>> inOrder) {
    if (acc)
      return _builder.CreateFMul(acc, in);
    return in;
  }

  Value *vecAccArith(Product &n, Value *acc1, Value* acc2) {
    return _builder.CreateFMul(acc1, acc2);
  }
  Value *vecAccArith(Sum &n, Value *acc1, Value* acc2) {
    return _builder.CreateFAdd(acc1, acc2);
  }
  Value *vecAccArith(WeightedSum &n, Value *acc1, Value* acc2) {
    // Note that the weightings have already been applied when individual
    // children were combined into the two accumulators, so combining the
    // accumulators is just a simple add
    return _builder.CreateFAdd(acc1, acc2);
  }
  
  bool createInputs(std::vector<
          std::pair<NodeReference, std::vector<std::pair<std::string, size_t>>>>
		    inputs) {

    while (inputs.size() > 0) {
      std::vector<
          std::pair<NodeReference, std::vector<std::pair<std::string, size_t>>>>
          deferredInputs;
      for (auto &in : inputs) {
        in.first->accept(
            *this,
            in.second.size()
                ? std::make_shared<std::vector<std::pair<std::string, size_t>>>(
                      in.second)
                : std::shared_ptr<
                      std::vector<std::pair<std::string, size_t>>>());
        if (tryOtherNode) {
          deferredInputs.push_back(in);
          tryOtherNode = false;
        }
      }
      if (deferredInputs.size() == inputs.size()) {
        // no more progress was made in this iteration, so we will try to
        // create the other inputs of the consumer of _n_ first
        return false;
      }
      inputs = deferredInputs;
    }
    return true;
  }
  template <class T> void emitArith(T &n, arg_t arg) {
    // Node might have been handled already by another node it shares a vector
    // with
    if (node2value.find(n.id()) != node2value.end())
      return;

    auto it = _vec.find(n.id());
    if (it == _vec.end()) {
      Value *out = nullptr;
      std::vector<
          std::pair<NodeReference, std::vector<std::pair<std::string, size_t>>>>
          inputs;
      
      for (int i = 0; i < getInputLength(n); i++) {
	inputs.push_back({getInput(n, i), {}});
      }
      
      if (!createInputs(inputs)) {
	tryOtherNode = true;
	return;
      }

      std::sort(inputs.begin(), inputs.end(), [this](auto& a, auto& b) {
	  auto ita = _vec.find(a.first->id());
	  auto itb = _vec.find(b.first->id());

	  if (ita == _vec.end()) {
	    return false;
	  } else if (itb == _vec.end()) {
	    return true;
	  } else {
	    return ita->second < itb->second;
	  }
	});
      for (int i = 0; i < getInputLength(n); i++) {
	out = handleScalarInput(n, out, inputs[i].first);
      }
      out->setName(n.id());
      node2value.insert({n.id(), {out, -1, 0}});
    } else {
      if (vecsWithOrder.find(it->second) != vecsWithOrder.end() && arg.get() == nullptr) {
	tryOtherNode = true;
	return;
      }
      std::unordered_map<size_t, std::vector<NodeReference>> histogramInputs;
      std::unordered_map<size_t, std::vector<NodeReference>> gaussInputs;
      std::unordered_map<size_t, std::vector<NodeReference>> scalarIn;
      std::unordered_set<size_t> directVec;
      struct isHistoVisitor : public BaseVisitor {
        void visitHistogram(Histogram &n, arg_t arg) { isHisto = true; }
        void visitGauss(Gauss &n, arg_t arg) { isHisto = false; }
        void visitProduct(Product &n, arg_t arg) { isHisto = false; }
        void visitSum(Sum &n, arg_t arg) { isHisto = false; }
        void visitWeightedSum(WeightedSum &n, arg_t arg) { isHisto = false; }
        bool isHisto = false;
      };
      isHistoVisitor histoCheck;

      struct isGaussVisitor : public BaseVisitor {
        void visitHistogram(Histogram &n, arg_t arg) { isGauss = false; }
        void visitGauss(Gauss &n, arg_t arg) { isGauss = true; }
        void visitProduct(Product &n, arg_t arg) { isGauss = false; }
        void visitSum(Sum &n, arg_t arg) { isGauss = false; }
        void visitWeightedSum(WeightedSum &n, arg_t arg) { isGauss = false; }
        bool isGauss = false;
      };
      isGaussVisitor gaussCheck;

      std::unordered_map<size_t, NodeReference> order;

      if (arg.get() == nullptr) {
	// There is no restriction on the order of the elements because all elements are extracted
	for (int i = 0; i < _vectors[it->second].size(); i++) {
	  order.insert({i, _vectors[it->second][i]});
	}
      } else {
	std::vector<NodeReference> flexible;
	auto* reqs = (std::vector<std::pair<std::string, size_t>>*) arg.get();
	for (auto& elem : _vectors[it->second]) {
	  bool set = false;
	  for (auto& req : *reqs) {
	    if (req.first == elem->id()) {
	      assert(order.find(req.second) == order.end());
	      order[req.second] = elem;
	      set = true;
	      break;
	    }
	  }
          if (!set) {
            flexible.push_back(elem);
          }
	}
	int i = 0;
        for (auto &e : flexible) {
          while (order.find(i) != order.end())
            i++;
          order[i] = e;
          i++;
        }
	std::vector<NodeReference> newRefVec;
      }

      size_t vectorWidth = 0;
      for (auto& l : order)
        vectorWidth = std::max(l.first + 1, vectorWidth);
      
      auto& directInputs = _directVecInputs[it->second];
      std::unordered_map<size_t, std::vector<std::pair<std::string, size_t>>> newReqs;
      
      for (auto& lane : order) {
	auto& in = lane.second;
	std::vector<NodeReference> laneGaussInputs;
	std::vector<NodeReference> laneHistoInputs;
	scalarIn.insert({lane.first, {}});
        // Used to check if we've already planned for an element of a vector to
        // be a direct op. In that case, we cannot use another element from that
        // same vector for the same lane, instead, the second needed value of
        // that vector will have to be extracted
        std::unordered_set<size_t> usedDirects;
        T *curNode = (T *)in.get();
        for (int j = 0; j < getInputLength(*curNode); j++) {
          NodeReference m = getInput(*curNode, j);

	  m->accept(histoCheck, {});

	  if (histoCheck.isHisto) {
	    laneHistoInputs.push_back(m);
	    continue;
	  }

          m->accept(gaussCheck, {});

	  if (gaussCheck.isGauss) {
	    laneGaussInputs.push_back(m);
	    continue;
	  }

	  
	  
          if (_vec.find(m->id()) == _vec.end()) {
	    // input is scalar
	    scalarIn[lane.first].push_back(m);
	  } else {
	    size_t vecId = _vec[m->id()];
	    if (directInputs.find(vecId) == directInputs.end() || usedDirects.find(vecId) != usedDirects.end()) {
              scalarIn[lane.first].push_back(m);
            } else {
	      usedDirects.insert(vecId);
	      newReqs[vecId].push_back({m->id(), lane.first});
            }
          }
        }

	// Sort so that closer values get into the same vector, improving locality

        std::sort(laneGaussInputs.begin(), laneGaussInputs.end(),
                  [](auto &a, auto &b) {
                    return ((Gauss *)a.get())->indexVar()->index() <
                           ((Gauss *)b.get())->indexVar()->index();
                  });

        std::sort(laneHistoInputs.begin(), laneHistoInputs.end(),
                  [](auto &a, auto &b) {
                    return ((Histogram *)a.get())->indexVar()->index() <
                           ((Histogram *)b.get())->indexVar()->index();
                  });

        histogramInputs.insert({lane.first, laneHistoInputs});
        gaussInputs.insert({lane.first, laneGaussInputs});
      }

      std::vector<
          std::pair<NodeReference, std::vector<std::pair<std::string, size_t>>>>
          inputs;

      for (auto &direct : newReqs) {
	inputs.push_back({_vectors[direct.first][0], direct.second});
	directVec.insert(direct.first);
      }

      for (auto& lane : scalarIn) {
	for (auto& node : lane.second) {
	  inputs.push_back({node, {}});
	}
      }
      if (!createInputs(inputs)) {
	tryOtherNode = true;
	return;
      }

      struct iteratorVector {
        iteratorVector(std::unordered_map<size_t, std::vector<NodeReference>> &histoIn) {
          for (auto &vec : histoIn) {
            iterators.insert({vec.first, vec.second.begin()});
            ends.insert({vec.first, vec.second.end()});
          }
        }

	bool isEnd() {
	  for (auto& it : iterators) {
	    if (it.second == ends[it.first])
	      return true;
	  }
	  return false;
	}

	void advance() {
	  for (auto& histIt : iterators) {
	    histIt.second++;
	  }
	}
        std::unordered_map<size_t,decltype(histogramInputs.begin()->second.begin())> iterators;
        std::unordered_map<size_t,decltype(histogramInputs.begin()->second.begin())> ends;
      };
      auto vecIt = iteratorVector(histogramInputs);
      for (;!vecIt.isEnd(); vecIt.advance()) {
	Value* histoLoad;
        std::unordered_map<size_t, NodeReference> histoNodeRefs;
        if (useGather) {
          std::unordered_map<size_t, Value *> inputOffsets;
          std::unordered_map<size_t, Value *> histogramArrays;
          for (auto &nodeRef : vecIt.iterators) {
            histoNodeRefs[nodeRef.first] = *(nodeRef.second);
            Histogram *histObject = (Histogram *)nodeRef.second->get();
            auto inputObject = histObject->indexVar();
            inputOffsets[nodeRef.first] = ConstantInt::getSigned(
                Type::getInt32Ty(_context), inputObject->index());
            histogramArrays[nodeRef.first] = getHistogramPtr(*histObject);
          }

          // For now we don't support avx512, so it's always a 4 wide input
          // address vector although we might actually only use 2
          assert(simdWidth == 2 || simdWidth == 4);
          Value *inputAddressVector =
              UndefValue::get(VectorType::get(Type::getInt32Ty(_context), 4));

          Value *histogramAddressVector = UndefValue::get(
              VectorType::get(IntegerType::get(_context, 64), simdWidth));

          for (auto &in : inputOffsets) {
            inputAddressVector = _builder.CreateInsertElement(
                inputAddressVector, in.second, in.first);
            histogramAddressVector = _builder.CreateInsertElement(
                histogramAddressVector,
                _builder.CreatePtrToInt(histogramArrays[in.first],
                                        IntegerType::get(_context, 64)),
                in.first);
          }

          Function *intGather = Intrinsic::getDeclaration(
              _module, llvm::Intrinsic::x86_avx2_gather_d_d);
          std::vector<Value *> intArgs;

          intArgs.push_back(
              UndefValue::get(VectorType::get(Type::getInt32Ty(_context), 4)));
          intArgs.push_back(_builder.CreatePointerCast(
              _in, PointerType::getUnqual(Type::getInt8Ty(_context))));
          intArgs.push_back(inputAddressVector);

          Value *mask = Constant::getNullValue(
              VectorType::get(Type::getInt32Ty(_context), 4));
          for (auto &in : inputOffsets) {
            mask = _builder.CreateInsertElement(
                mask, Constant::getAllOnesValue(Type::getInt32Ty(_context)),
                in.first);
          }

          intArgs.push_back(mask);
          intArgs.push_back(ConstantInt::getSigned(Type::getInt8Ty(_context),
                                                   sizeof(int32_t)));

          Value *inputVector = _builder.CreateCall(intGather, intArgs);

          if (simdWidth == 2) {
            inputVector = _builder.CreateShuffleVector(
                inputVector,
                UndefValue::get(VectorType::get(Type::getInt32Ty(_context), 4)),
                {0, 1});
          }

          auto doubleSizeSplat = _builder.CreateVectorSplat(
              simdWidth,
              ConstantInt::get(IntegerType::get(_context, 64), sizeof(double)));
          auto zextInputs = _builder.CreateZExt(
              inputVector,
              VectorType::get(IntegerType::get(_context, 64), simdWidth));
          auto scaledInput = _builder.CreateMul(doubleSizeSplat, zextInputs);
          auto histoLoadAddresses =
              _builder.CreateAdd(scaledInput, histogramAddressVector);

          Function *doubleGather;

          if (simdWidth == 2)
            doubleGather = Intrinsic::getDeclaration(
                _module, llvm::Intrinsic::x86_avx2_gather_q_pd);
          else
            doubleGather = Intrinsic::getDeclaration(
                _module, llvm::Intrinsic::x86_avx2_gather_q_pd_256);
          std::vector<Value *> doubleArgs;

          doubleArgs.push_back(UndefValue::get(
              VectorType::get(Type::getDoubleTy(_context), simdWidth)));
          doubleArgs.push_back(ConstantPointerNull::get(
              PointerType::getUnqual(Type::getInt8Ty(_context))));
          doubleArgs.push_back(histoLoadAddresses);

          Value *doubleMask = Constant::getNullValue(
              VectorType::get(Type::getDoubleTy(_context), simdWidth));
          for (auto &in : inputOffsets) {
            doubleMask = _builder.CreateInsertElement(
                doubleMask,
                Constant::getAllOnesValue(Type::getDoubleTy(_context)),
                in.first);
          }

          doubleArgs.push_back(doubleMask);
          doubleArgs.push_back(
              ConstantInt::getSigned(Type::getInt8Ty(_context), 1));

          histoLoad = _builder.CreateCall(doubleGather, doubleArgs);
        } else {
          histoLoad = UndefValue::get(
              VectorType::get(Type::getDoubleTy(_context), vectorWidth));
          bool binaryPossible = true;
          if (selectBinary) {
            for (auto &nodeRef : vecIt.iterators) {
              Histogram *histObject = (Histogram *)nodeRef.second->get();
              if (histObject->buckets()->size() != 2)
                binaryPossible = false;
            }
          }

          if (selectBinary && binaryPossible) {
            Value* histoBounds = UndefValue::get(
                VectorType::get(Type::getInt32Ty(_context), vectorWidth));

            Value* lowerVals = UndefValue::get(
                VectorType::get(Type::getDoubleTy(_context), vectorWidth));
            Value* upperVals = UndefValue::get(
                VectorType::get(Type::getDoubleTy(_context), vectorWidth));

            Value* compareInputs =
                UndefValue::get(VectorType::get(Type::getInt32Ty(_context), vectorWidth));

            for (auto &nodeRef : vecIt.iterators) {
              histoNodeRefs[nodeRef.first] = *(nodeRef.second);
              Histogram *histObject = (Histogram *)nodeRef.second->get();
	      auto& buckets = *histObject->buckets();
              histoBounds = _builder.CreateInsertElement(
                  histoBounds,
                  ConstantInt::getSigned(Type::getInt32Ty(_context),
                                         buckets[0].upperBound),
                  nodeRef.first);

              if (input2value.find(histObject->indexVar()->id()) == input2value.end()) {
                histObject->indexVar()->accept(*this, {});
              }

              compareInputs = _builder.CreateInsertElement(
                  compareInputs, input2value[histObject->indexVar()->id()],
                  nodeRef.first);

              lowerVals = _builder.CreateInsertElement(
                  lowerVals,
                  ConstantFP::get(Type::getDoubleTy(_context), buckets[0].value),
                  nodeRef.first);
	      
              upperVals = _builder.CreateInsertElement(
                  upperVals,
                  ConstantFP::get(Type::getDoubleTy(_context), buckets[1].value),
                  nodeRef.first);
            }

            auto lt = _builder.CreateICmpSLT(compareInputs, histoBounds);
	    histoLoad = _builder.CreateSelect(lt, lowerVals, upperVals);

          } else {
            for (auto &nodeRef : vecIt.iterators) {
              histoNodeRefs[nodeRef.first] = *(nodeRef.second);
              Histogram *histObject = (Histogram *)nodeRef.second->get();
              histObject->accept(*this, {});
              histoLoad = _builder.CreateInsertElement(
                  histoLoad, node2value[histObject->id()].val, nodeRef.first);
            }
          }
        }
        size_t newId = _vectors.size();
        std::vector<NodeReference> newRefVec;
        for (auto &ref : histoNodeRefs) {
          newRefVec.push_back(ref.second);
        }
        _vectors.push_back(newRefVec);
        std::string histVecName = "";
        for (auto &ref : histoNodeRefs) {
          histVecName += ref.second->id() + "_";
          node2value[ref.second->id()] = {histoLoad,
                                          static_cast<int>(ref.first), newId};
          newReqs[newId].push_back({ref.second->id(), ref.first});
        }
        //inputVector->setName(histVecName + "idx");
        histoLoad->setName(histVecName);
        directVec.insert(newId);
      }

      std::vector<size_t> laneIdxs;
      for (auto& lane : gaussInputs) {
	laneIdxs.push_back(lane.first);
      }

      std::sort(laneIdxs.begin(), laneIdxs.end(), [&](size_t a, size_t b) {
        return gaussInputs[a].size() > gaussInputs[b].size();
      });

      std::vector<size_t> gaussVectorIdx(laneIdxs.size(), 0);
      Value* biggestLane = nullptr;
      
      for (int i = 0; gaussInputs[laneIdxs[0]].size()-i != gaussInputs[laneIdxs[1]].size(); i++, gaussVectorIdx[0]++) {
	auto& gaussNode = gaussInputs[laneIdxs[0]][i];
	gaussNode->accept(*this, {});
	auto& gaussVal = node2value[gaussNode->id()].val;
	biggestLane = scalarArith(n, biggestLane, gaussVal, gaussNode->id());
      }

      Value* gaussAccVec = nullptr;
      if (biggestLane) {
        gaussAccVec = UndefValue::get(
            VectorType::get(Type::getDoubleTy(_context), laneIdxs[0]+1));
        gaussAccVec = _builder.CreateInsertElement(gaussAccVec, biggestLane, laneIdxs[0]);
      }
      size_t activeLanesInAcc = 1;
      size_t accWidth = laneIdxs[0] + 1;
      // now new gauss vectors can be created an accumulated onto gaussAccVec
      for (int i = 1; i < laneIdxs.size(); i++) {
        size_t gaussiansToHandle =
            gaussInputs[laneIdxs[i]].size() -
            (i < laneIdxs.size() - 1 ? gaussInputs[laneIdxs[i + 1]].size() : 0);

        size_t newAccWidth = 0;
        for (int j = 0; j <= i; j++) {
          newAccWidth = std::max(laneIdxs[j], newAccWidth);
        }
        newAccWidth++;
	
	if (gaussAccVec && (gaussiansToHandle > 0 || i+1 == laneIdxs.size())) {
	  if (newAccWidth > accWidth) {

            Value* maskVec = UndefValue::get(
                VectorType::get(Type::getInt32Ty(_context), newAccWidth));

	    for (int j = 0; j < activeLanesInAcc; j++) {
              maskVec = _builder.CreateInsertElement(
                  maskVec,
                  ConstantInt::getSigned(Type::getInt32Ty(_context),
                                         laneIdxs[j]),
                  laneIdxs[j]);
            }

	    for (int  j = activeLanesInAcc; j <= i; j++) {
              maskVec = _builder.CreateInsertElement(
                  maskVec,
                  ConstantInt::getSigned(Type::getInt32Ty(_context), accWidth),
                  laneIdxs[j]);
            }

            Value* blendVec = UndefValue::get(
                VectorType::get(Type::getDoubleTy(_context), accWidth));
            blendVec =
	      _builder.CreateInsertElement(blendVec, getNeutralValue<T>(), uint64_t(0));
	    gaussAccVec = _builder.CreateShuffleVector(gaussAccVec, blendVec, maskVec);
          } else {
	    for (int j = activeLanesInAcc; j <= i; j++) {
	      gaussAccVec = _builder.CreateInsertElement(gaussAccVec, getNeutralValue<T>(), laneIdxs[j]);
	    }
	  }
	  activeLanesInAcc = i+1;
	  accWidth = newAccWidth;
        } else if (gaussiansToHandle > 0) {
          accWidth = newAccWidth;
        }

        for (int j = 0; j < gaussiansToHandle; j++) {
	  std::vector<NodeReference> gaussNodes;
          Value* constFactor = UndefValue::get(
              VectorType::get(Type::getDoubleTy(_context), newAccWidth));
	  
          Value* constDivisor = UndefValue::get(
              VectorType::get(Type::getDoubleTy(_context), newAccWidth));
	  
          Value* mean = UndefValue::get(
              VectorType::get(Type::getDoubleTy(_context), newAccWidth));
	  
          Value* observation = UndefValue::get(
              VectorType::get(Type::getDoubleTy(_context), newAccWidth));
	  std::vector<std::pair<std::string, size_t>> gaussVecPositions;
	  std::unordered_map<size_t, NodeReference> accPositions;
          for (int k = 0; k <= i; k++) {
	    size_t laneNo = laneIdxs[k];
	    accPositions[laneNo] = order[laneNo];
            Gauss* nr = (Gauss*) gaussInputs[laneNo][gaussVectorIdx[k]].get();
	    double fac = 1 / (std::sqrt(2 * M_PI * nr->stddev() * nr->stddev()));
            constFactor = _builder.CreateInsertElement(
                constFactor,
                ConstantFP::get(Type::getDoubleTy(_context), fac), laneNo);
            constDivisor = _builder.CreateInsertElement(
                constDivisor,
                ConstantFP::get(Type::getDoubleTy(_context),
                                -2.0 * nr->stddev() * nr->stddev()),
                laneNo);

	    mean = _builder.CreateInsertElement(
                mean,
                ConstantFP::get(Type::getDoubleTy(_context),
				nr->mean()),
                laneNo);

            if (input2value.find(nr->indexVar()->id()) == input2value.end()) {
              nr->indexVar()->accept(*this, {});
            }

            observation = _builder.CreateInsertElement(
                observation,
                _builder.CreateSIToFP(input2value[nr->indexVar()->id()],
                                      Type::getDoubleTy(_context)),
                laneNo);
            gaussVecPositions.push_back({nr->id(), laneNo});
            gaussVectorIdx[k]++;
	  }
	  auto normed = _builder.CreateFSub(observation, mean);
	  auto squared = _builder.CreateFMul(normed, normed);
	  
	  auto division = _builder.CreateFDiv(squared, constDivisor);

          auto expFunc = Intrinsic::getDeclaration(
              _module, llvm::Intrinsic::exp,
              {VectorType::get(Type::getDoubleTy(_context), newAccWidth)});

          auto expRes = _builder.CreateCall(expFunc, {division});
          auto density = _builder.CreateFMul(expRes, constFactor);
          gaussAccVec = vecArith(n, gaussAccVec, density, accPositions, gaussVecPositions);
        }
      }

      bool hasScalarInputs = false;
      std::unordered_map<size_t, Value *> serialInputs;
      // First emit all reductions of inputs which do not come in a vector
      for (auto &lane : order) {
        T *curNode = (T *)lane.second.get();

        Value *aggSerIn = nullptr;

        while (vecIt.iterators[lane.first] != vecIt.ends[lane.first]) {
          auto &hist = vecIt.iterators[lane.first];
          // Note: this accept can't fail, since it's a scalar histogram, thus
          // has neither inputs nor order
          (*hist)->accept(*this, {});
          aggSerIn = handleScalarInput(*curNode, aggSerIn, *hist);
          hist++;
        }

        for (auto &scalarRef : scalarIn[lane.first]) {
          aggSerIn = handleScalarInput(*curNode, aggSerIn, scalarRef);
        }

        serialInputs[lane.first] = aggSerIn;
	if (aggSerIn)
	  hasScalarInputs=true;
      }
      Value *out = nullptr;

      if (hasScalarInputs) {
        out = UndefValue::get(
            VectorType::get(Type::getDoubleTy(_context), vectorWidth));

        for (auto &lane : order) {
          out = _builder.CreateInsertElement(out,
                                             serialInputs[lane.first]
                                                 ? serialInputs[lane.first]
                                                 : getNeutralValue<T>(),
                                             lane.first);
        }
      }
      if (out && gaussAccVec)
	out = vecAccArith(n, out, gaussAccVec);
      else if (gaussAccVec)
	out = gaussAccVec;
      
      for (auto& direct : directVec) {
        Value *directVecVal = node2value[_vectors[direct][0]->id()].val;
        for (auto &lane : order) {
          bool needed = false;

          for (auto &providedVal : newReqs[direct]) {
            if (lane.first == providedVal.second) {
              needed = true;
              break;
            }
          }
          if (!needed) {
            directVecVal = _builder.CreateInsertElement(
                directVecVal,getNeutralValue<T>(),
                lane.first);
          }
        }

	size_t inWidth = ((VectorType *)directVecVal->getType())->getElementCount().Min;
        if (inWidth != vectorWidth) {
          std::vector<uint32_t> indices;
	  
	  for (int i = 0; i < vectorWidth; i++) {
	    if (order.find(i) != order.end()) {
	      indices.push_back(i);
	    } else {
	      indices.push_back(inWidth);
	    }
	  }
          directVecVal = _builder.CreateShuffleVector(
              directVecVal, UndefValue::get(directVecVal->getType()), indices);
        }

        out = vecArith(n, out, directVecVal, order, newReqs[direct]);
      }

      std::string valName = "";
      for (auto& lane : order) {
	valName += lane.second->id() + "_";
        node2value.insert(
			  {lane.second->id(), {out, static_cast<int>(lane.first), it->second}});
      }
      out->setName(valName);
    }
  }
  std::unordered_map<std::string, size_t>& _vec;
  std::unordered_map<size_t, std::unordered_set<size_t>>& _directVecInputs;
  std::vector<std::vector<NodeReference>>& _vectors;
  std::unordered_set<size_t> vecsWithOrder;
  bool tryOtherNode = false;
  Value *_in;
  Function *_func;
  LLVMContext &_context;
  IRBuilder<> &_builder;
  Module *_module;
  std::unordered_map<std::string, irVal> node2value;
  std::unordered_map<std::string, Value *> input2value;
  unsigned simdWidth;
  Value* constantZero;
  Value* constantOne;
};
