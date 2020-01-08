#include <unordered_set>
#include <graph-ir/GraphIRNode.h>
#include "llvm/IR/IRBuilder.h"
#include "transform/BaseVisitor.h"
#include <unordered_map>
#include <set>
#include <iostream>

using namespace llvm;

struct irVal {
  Value* val;
  int pos;
  size_t vec;
};

class IREmitter : public BaseVisitor {
public:
  IREmitter(
      std::unordered_map<std::string, size_t> &vec,
      std::unordered_map<size_t, std::unordered_map<size_t, std::vector<std::string>>>& directVecInputs,
      std::vector<std::vector<NodeReference>>& vectors, Value *in,
      Function *func, LLVMContext &context, IRBuilder<> &builder,
      Module *module, unsigned width);

  void visitInputvar(InputVar &n, arg_t arg) override;

  void visitHistogram(Histogram &n, arg_t arg) override;

  void visitProduct(Product &n, arg_t arg) override;
  void visitSum(Sum &n, arg_t arg) override;
  void visitWeightedSum(WeightedSum &n, arg_t arg) override;

  std::unordered_map<std::string, irVal> getNodeMap();

private:
  Value* getHistogramPtr(Histogram& n);
  template <class T> double getNeutralValue();
  template <> double getNeutralValue<WeightedSum>() {return 0.0;}
  template <> double getNeutralValue<Sum>() {return 0.0;}
  template <> double getNeutralValue<Product>() {return 1.0;}

  size_t getInputLength(WeightedSum &n) {
    return n.addends()->size();
  }

  size_t getInputLength(Sum &n) {
    return n.addends()->size();
  }

  size_t getInputLength(Product &n) {
    return n.multiplicands()->size();
  }

  NodeReference getInput(WeightedSum& n, int pos) {
    return (*n.addends())[pos].addend;
  }
  NodeReference getInput(Sum& n, int pos) {
    return (*n.addends())[pos];
  }
  NodeReference getInput(Product& n, int pos) {
    return (*n.multiplicands())[pos];
  }

  Value *scalarArith(WeightedSum &n, Value *acc, Value *in, std::string id) {
    double weight = 0.0;
    for (auto& addend : *n.addends()) {
      if (addend.addend->id() == id) {
	weight = addend.weight;
	break;
      }
    }
    //assert(weight > 0.0);
    auto mul = _builder.CreateFMul(
        ConstantFP::get(Type::getDoubleTy(_context), weight), in);
    return _builder.CreateFAdd(acc, mul);
  }
  Value *scalarArith(Sum &n, Value *acc, Value *in, std::string id) {
    return _builder.CreateFAdd(acc, in);
  }
  
  Value *scalarArith(Product &n, Value *acc, Value *in, std::string id) {
    return _builder.CreateFMul(acc, in);
  }

  template <class T>
  Value *handleScalarInput(T &n, Value *acc, NodeReference in) {
    if (node2value.find(in->id()) == node2value.end())
      in->accept(*this, {});
    auto inInfo = node2value[in->id()];
    Value *inVal;
    if (inInfo.pos == -1)
      inVal = inInfo.val;
    else
      inVal = _builder.CreateExtractElement(inInfo.val, inInfo.pos);
    return scalarArith(n, acc, inVal, in->id());
  }

  Value *vecArith(WeightedSum &n, Value *acc, Value* in, size_t inId, size_t width) {
    std::vector<Value *> weights;

    for (int i = 0; i < width; i++) {
      std::string inputName;
      if (i < _vectors[inId].size())
	inputName = _vectors[inId][i]->id();
      
      auto inputs = ((WeightedSum *)_vectors[_vec[n.id()]][i].get())->addends();
      double weight = 0.0;
      for (auto &w : *inputs) {
        if (w.addend->id() == inputName) {
          weight = w.weight;
        }
      }
      // Note: it may be, that weight is not set, because the value in slot _i_
      // of _inId_ is not actually used, thus we cannot find any input for
      // inputName, which is either "" or an id that is used by other values. We
      // can choose any weight then, because the i-th value in in _in_ is 0.0
      // (i.e. neutral wrt addition) anyway, so we just leave it at 0.0
      weights.push_back(ConstantFP::get(Type::getDoubleTy(_context), weight));
    }

    Value *weightVec = _builder.CreateVectorSplat(simdWidth, weights[0]);

    for (int i = 1; i < width; i++) {
      weightVec = _builder.CreateInsertElement(weightVec, weights[i], i);
    }
    auto mul = _builder.CreateFMul(in, weightVec);
    return _builder.CreateFAdd(acc, mul);
  }

  Value *vecArith(Sum &n, Value *acc, Value* in, size_t inId, size_t width) {
    return _builder.CreateFAdd(acc, in);
  }

  Value *vecArith(Product &n, Value *acc, Value* in, size_t inId, size_t width) {
    return _builder.CreateFMul(acc, in);
  }
  
  template <class T> void emitArith(T &n, arg_t arg) {
    // Node might have been handled already by another node it shares a vector
    // with
    if (node2value.find(n.id()) != node2value.end())
      return;

    auto it = _vec.find(n.id());
    if (it == _vec.end()) {
      Value *out = ConstantFP::get(Type::getDoubleTy(_context), getNeutralValue<T>());
      for (int i = 0; i < getInputLength(n); i++) {
	out = handleScalarInput(n, out, getInput(n, i));
      }
      out->setName(n.id());
      node2value.insert({n.id(), {out, -1, 0}});
    } else {
      // scalar histo, non histo scalar, direct vec, extract vec
      std::vector<std::vector<NodeReference>> histogramInputs;
      // first is vector id, second is lane in this vector
      std::vector<std::set<std::pair<size_t, unsigned>>> extractVec;
      std::vector<std::vector<NodeReference>> scalarIn;
      std::unordered_set<size_t> directVec;
      struct isHistoVisitor : public BaseVisitor {
        void visitHistogram(Histogram &n, arg_t arg) { isHisto = true; }

        void visitProduct(Product &n, arg_t arg) { isHisto = false; }
        void visitSum(Sum &n, arg_t arg) { isHisto = false; }
        void visitWeightedSum(WeightedSum &n, arg_t arg) { isHisto = false; }
        bool isHisto = false;
      };
      isHistoVisitor histoCheck;

      std::vector<NodeReference> order;
      order.resize(_vectors[it->second].size());

      
      if (arg.get() == nullptr) {
	// There is no restriction on the order of the elements because all elements are extracted
	order = _vectors[it->second];
      } else {
	std::unordered_set<size_t> used;
	std::vector<NodeReference> flexible;
	auto* reqs = (std::vector<std::pair<std::string, size_t>>*) arg.get();
	for (auto& elem : _vectors[it->second]) {
	  bool set = false;
	  for (auto& req : *reqs) {
	    if (req.first == elem->id()) {
	      order[req.second] = elem;
	      assert(used.find(req.second) == used.end());
	      used.insert(req.second);
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
          while (used.find(i) != used.end())
            i++;
          order[i] = e;
	  i++;
        }
      }

      auto& directInputs = _directVecInputs[it->second];
      std::unordered_map<size_t, std::vector<std::pair<std::string, size_t>>> newReqs;
      
      for (int i = 0; i < order.size(); i++) {
	auto& in = order[i];
        // Used to check if we've already planned for an element of a vector to
        // be a direct op In that case, we cannot use another element from that
        // same vector for the same lane, instead, the second needed value of
        // that vector will have to be extracted
        std::unordered_set<size_t> usedDirects;
        T *curNode = (T *)in.get();
	histogramInputs.push_back({});
	extractVec.push_back({});
	scalarIn.push_back({});
        for (int j = 0; j < getInputLength(*curNode); j++) {
          NodeReference m = getInput(*curNode, j);

	  m->accept(histoCheck, {});

	  if (histoCheck.isHisto) {
	    histogramInputs.back().push_back(m);
	    continue;
	  }
	  
          if (_vec.find(m->id()) == _vec.end()) {
	    // input is scalar
            m->accept(*this, {});
	    scalarIn.back().push_back(m);
	  } else {
	    size_t vecId = _vec[m->id()];
	    if (directInputs.find(vecId) == directInputs.end() || usedDirects.find(vecId) != usedDirects.end()) {
              m->accept(*this, {});
              if (n.id() == "8" || n.id() == "26")
                std::cout << "hit" << std::endl;
              extractVec.back().insert({vecId, node2value[m->id()].pos});
            } else {
	      usedDirects.insert(vecId);
	      newReqs[vecId].push_back({m->id(), i});
            }
          }
        }
      }

      // now that all position requirements for direct input vectors are collected, they can be created
      for (auto &direct : newReqs) {
        _vectors[direct.first][0]->accept(
            *this,
            std::make_shared<std::vector<std::pair<std::string, size_t>>>(
                direct.second));
	directVec.insert(direct.first);
      }

      struct iteratorVector {
        iteratorVector(std::vector<std::vector<NodeReference>> &histoIn) {
          for (auto &vec : histoIn) {
            iterators.push_back(vec.begin());
            ends.push_back(vec.end());
          }
        }

	bool isEnd() {
	  for (int i = 0; i < ends.size(); i++) {
	    if (iterators[i] == ends[i])
	      return true;
	  }
	  return false;
	}

	void advance() {
	  for (auto& histIt : iterators) {
	    histIt++;
	  }
	}
        std::vector<decltype(histogramInputs[0].begin())> iterators;
        std::vector<decltype(histogramInputs[0].begin())> ends;
      };
      
      auto vecIt = iteratorVector(histogramInputs);
      for (; !vecIt.isEnd(); vecIt.advance()) {
        std::vector<Value*> inputOffsets;
        std::vector<Value*> histogramArrays;
	std::vector<NodeReference> histoNodeRefs;
        for (auto &nodeRef : vecIt.iterators) {
	  histoNodeRefs.push_back(*nodeRef);
          Histogram *histObject = (Histogram *)nodeRef->get();
          auto inputObject = histObject->indexVar();
          inputOffsets.push_back(ConstantInt::getSigned(
              Type::getInt32Ty(_context), inputObject->index()));
          histogramArrays.push_back(getHistogramPtr(*histObject));
        }

        // For now we don't support avx512, so it's always a 4 wide input
        // address vector although we might actually only use 2
	assert(simdWidth == 2 || simdWidth == 4);
        Value *inputAddressVector =
            _builder.CreateVectorSplat(4, inputOffsets[0]);

        Value *histogramAddressVector =
	  _builder.CreateVectorSplat(simdWidth, _builder.CreatePtrToInt(histogramArrays[0], IntegerType::get(_context, 64)));

        for (int i = 1; i < order.size(); i++) {
          inputAddressVector = _builder.CreateInsertElement(
              inputAddressVector, inputOffsets[i], i);
          histogramAddressVector = _builder.CreateInsertElement(
              histogramAddressVector, _builder.CreatePtrToInt(histogramArrays[i], IntegerType::get(_context, 64)), i);
        }

	Function* intGather = Intrinsic::getDeclaration(_module,llvm::Intrinsic::x86_avx2_gather_d_d);
	std::vector<Value*> intArgs;

	intArgs.push_back(UndefValue::get(VectorType::get(Type::getInt32Ty(_context), 4)));
	intArgs.push_back(_builder.CreatePointerCast(_in, PointerType::getUnqual(Type::getInt8Ty(_context))));
	intArgs.push_back(inputAddressVector);
        intArgs.push_back(Constant::getAllOnesValue(
            VectorType::get(Type::getInt32Ty(_context), 4)));
        intArgs.push_back(
            ConstantInt::getSigned(Type::getInt8Ty(_context), sizeof(int32_t)));

	Value* inputVector = _builder.CreateCall(intGather, intArgs);

	if (simdWidth == 2) {
          inputVector = _builder.CreateShuffleVector(
              inputVector,
              UndefValue::get(VectorType::get(Type::getInt32Ty(_context), 4)),
              {0, 1});
        }

        auto scaledInput = _builder.CreateMul(
            _builder.CreateZExt(
                inputVector,
                VectorType::get(IntegerType::get(_context, 64), simdWidth)),
            _builder.CreateVectorSplat(
                simdWidth, ConstantInt::get(IntegerType::get(_context, 64),
                                            sizeof(double))));
        auto histoLoadAddresses = _builder.CreateAdd(scaledInput, histogramAddressVector);

	Function* doubleGather;

	if (simdWidth == 2)
	  doubleGather = Intrinsic::getDeclaration(_module,llvm::Intrinsic::x86_avx2_gather_q_pd);
	else
	  doubleGather = Intrinsic::getDeclaration(_module,llvm::Intrinsic::x86_avx2_gather_q_pd_256);
	std::vector<Value*> doubleArgs;
	
	doubleArgs.push_back(UndefValue::get(VectorType::get(Type::getDoubleTy(_context), simdWidth)));
        doubleArgs.push_back(ConstantPointerNull::get(
            PointerType::getUnqual(Type::getInt8Ty(_context))));
        doubleArgs.push_back(histoLoadAddresses);
        doubleArgs.push_back(Constant::getAllOnesValue(
            VectorType::get(Type::getDoubleTy(_context), simdWidth)));
        doubleArgs.push_back(
            ConstantInt::getSigned(Type::getInt8Ty(_context), 1));

	auto histoLoad = _builder.CreateCall(doubleGather, doubleArgs);
        size_t newId = _vectors.size();
        _vectors.push_back(histoNodeRefs);
        std::string histVecName = "";
        for (int i = 0; i < order.size(); i++) {
          histVecName += histoNodeRefs[i]->id() + "_";
          node2value.insert({histoNodeRefs[i]->id(), {histoLoad, i, newId}});
        }
	inputVector->setName(histVecName+"idx");
        histoLoad->setName(histVecName);
        directVec.insert(newId);
      }
      
      std::vector<Value *> serialInputs;
      // First emit all reductions of inputs which do not come in a vector
      for (int i = 0; i < order.size(); i++) {
        T *curNode = (T *)order[i].get();

        Value *aggSerIn =
            ConstantFP::get(Type::getDoubleTy(_context), getNeutralValue<T>());

        while (vecIt.iterators[i] != vecIt.ends[i]) {
          auto &hist = vecIt.iterators[i];
          aggSerIn = handleScalarInput(*curNode, aggSerIn, *hist);
          hist++;
        }

        for (auto &scalarRef : scalarIn[i]) {
          aggSerIn = handleScalarInput(*curNode, aggSerIn, scalarRef);
        }

        for (auto &extract : extractVec[i]) {
          aggSerIn =
              handleScalarInput(*curNode, aggSerIn, _vectors[extract.first][extract.second]);
        }
        serialInputs.push_back(aggSerIn);
      }
      Value *out = _builder.CreateVectorSplat(simdWidth, serialInputs[0]);

      for (int i = 1; i < order.size(); i++) {
        out = _builder.CreateInsertElement(out, serialInputs[i], i);
      }
      
      for (auto& direct : directVec) {
	if (newReqs.find(direct) == newReqs.end() || newReqs[direct].size() == order.size()) {
          out = vecArith(n, out, node2value[_vectors[direct][0]->id()].val, direct, order.size());
        } else {
	  // not all _values_ in direct are needed by _out_
	  std::cout << "half extract in " << n.id() << std::endl;
	  Value* directVecVal = node2value[_vectors[direct][0]->id()].val;
	  for (int i = 0; i < order.size(); i++) {
	    bool needed = false;
            if (i < _vectors[direct].size()) {
              auto &direcLane = _vectors[direct][i];
              for (auto &neededVal : newReqs[direct]) {
                if (direcLane->id() == neededVal.first) {
                  needed = true;
                  break;
                }
              }
            }
            if (!needed) {
              directVecVal = _builder.CreateInsertElement(
                  directVecVal,
                  ConstantFP::get(Type::getDoubleTy(_context),
                                  getNeutralValue<T>()),
                  i);
            }
          }
	  out = vecArith(n, out, directVecVal, direct, order.size());
	}
	
      }
      std::string valName = "";
      for (int i = 0; i < order.size(); i++) {
	valName += _vectors[it->second][i]->id() + "_";
        node2value.insert(
            {_vectors[it->second][i]->id(), {out, i, it->second}});
      }
      out->setName(valName);
    }
  }
  std::unordered_map<std::string, size_t>& _vec;
  std::unordered_map<size_t, std::unordered_map<size_t, std::vector<std::string>>>& _directVecInputs;
  std::vector<std::vector<NodeReference>>& _vectors;
  Value *_in;
  Function *_func;
  LLVMContext &_context;
  IRBuilder<> &_builder;
  Module *_module;
  std::unordered_map<std::string, irVal> node2value;
  std::unordered_map<std::string, Value *> input2value;
  unsigned simdWidth;
};
