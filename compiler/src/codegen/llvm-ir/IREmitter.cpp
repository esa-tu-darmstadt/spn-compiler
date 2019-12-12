#include "codegen/llvm-ir/IREmitter.h"

IREmitter::IREmitter(
      std::unordered_map<std::string, size_t> &vec,
      std::unordered_map<size_t, std::unordered_set<size_t>>& directVecInputs,
      std::unordered_map<size_t, std::vector<NodeReference>>& vectors, Value *in,
      Function *func, LLVMContext &context, IRBuilder<> &builder,
      Module *module, unsigned width)
    : _vec(vec), _directVecInputs(directVecInputs), _vectors(vectors), _in(in),
      _func(func), _context(context), _builder(builder), _module(module),
      simdWidth(width) {}

void IREmitter::visitInputvar(InputVar &n, arg_t arg) {
  // Node might have been handled already by another node it shares a vector
  // with
  if (input2value.find(n.id()) != input2value.end())
    return;
  
  // Not vectorized at the moment
  auto addr = _builder.CreateGEP(
      _in, {ConstantInt::getSigned(Type::getInt64Ty(_context), n.index())});
  auto val = _builder.CreateLoad(Type::getInt32Ty(_context), addr);
  input2value.insert({n.id(), val});
}

void IREmitter::visitHistogram(Histogram &n, arg_t arg) {
  // TODO Vectorized version generated, not emitted atm, use avx gather and
  // constantarray (when sensible)

  // Node might have been handled already by another node it shares a vector
  // with
  if (node2value.find(n.id()) != node2value.end())
    return;
  
  if (input2value.find(n.indexVar()->id()) == input2value.end()) {
    n.indexVar()->accept(*this, {});
  }
  
  auto in = input2value[n.indexVar()->id()];
  auto &buckets = *n.buckets();
  if (buckets.size() > 10000) {
    auto inBlock = _builder.GetInsertBlock();
    auto exit = BasicBlock::Create(_context, "exit" + n.id(), _func);
    _builder.SetInsertPoint(exit);
    auto out = _builder.CreatePHI(Type::getDoubleTy(_context), buckets.size());
    node2value.insert({n.id(), {out, -1, 0}});
    _builder.SetInsertPoint(inBlock);
    std::vector<BasicBlock *> matches;
    for (auto &b : buckets) {
      auto ge = _builder.CreateICmpSGE(
          in, ConstantInt::getSigned(Type::getInt32Ty(_context), b.lowerBound));
      auto lt = _builder.CreateICmpSLT(
          in, ConstantInt::getSigned(Type::getInt32Ty(_context), b.upperBound));
      auto cmp = _builder.CreateAnd(ge, lt);
      auto inRange = BasicBlock::Create(_context,
                                        "matched " + n.id() + ": " +
                                            std::to_string(b.lowerBound) + "-" +
                                            std::to_string(b.upperBound),
                                        _func);
      matches.push_back(inRange);
      auto nextIf = BasicBlock::Create(_context,
                                       "nextIf " + n.id() + ": " +
                                           std::to_string(b.lowerBound) + "-" +
                                           std::to_string(b.upperBound),
                                       _func);
      _builder.CreateCondBr(cmp, inRange, nextIf);
      _builder.SetInsertPoint(inRange);
      _builder.CreateBr(exit);
      out->addIncoming(ConstantFP::get(Type::getDoubleTy(_context), b.value),
                       inRange);
      _builder.SetInsertPoint(nextIf);
    }

    // The input var is outside the range of the histogram, so we abort
    auto exitFuncType = FunctionType::get(Type::getVoidTy(_context),
                                          Type::getInt32Ty(_context), true);
    auto exitFunc = _module->getOrInsertFunction("exit", exitFuncType);
    _builder.CreateCall(exitFunc,
                        ConstantInt::get(Type::getInt32Ty(_context), 1));
    _builder.CreateRetVoid();
    _builder.SetInsertPoint(exit);
  } else {
    std::vector<Constant*> values;
    size_t max_index = 0;
    for(auto& b : *n.buckets()){
        for(int i=0; i < (b.upperBound - b.lowerBound); ++i){
            values.push_back(ConstantFP::get(Type::getDoubleTy(_context), b.value));
        }
        max_index = (b.upperBound > max_index) ? b.upperBound : max_index;
    }
    assert(max_index == values.size());
    auto arrayType = ArrayType::get(Type::getDoubleTy(_context), values.size());
    auto initializer = ConstantArray::get(arrayType, values);
    auto globalArray = new GlobalVariable(*_module, arrayType, true, GlobalValue::InternalLinkage,
                                          initializer);
    auto address = _builder.CreateGEP(
        globalArray, {ConstantInt::get(IntegerType::get(_context, 32), 0), in});
    auto out = _builder.CreateLoad(address);
    node2value.insert({n.id(), {out, -1, 0}});
  }
}

void IREmitter::visitProduct(Product &n, arg_t arg) {
  // Node might have been handled already by another node it shares a vector
  // with
  if (node2value.find(n.id()) != node2value.end())
    return;
  auto it = _vec.find(n.id());
  if (it == _vec.end()) {
    Value *out = ConstantFP::get(Type::getDoubleTy(_context), 1.0);
    for (auto &m : *n.multiplicands()) {
      if (node2value.find(m->id()) == node2value.end())
	m->accept(*this, {});
      auto in = node2value[m->id()];
      Value *inVal;
      if (in.pos == -1)
        inVal = in.val;
      else
        inVal = _builder.CreateExtractElement(in.val, in.pos);
      out = _builder.CreateFMul(out, inVal);
    }
    node2value.insert({n.id(), {out, -1, 0}});
  } else {
    // operation is vectorized
    std::vector<Value *> serialInputs;
    // First emit all multiplications of inputs which do not come in a vector
    for (int i = 0; i < simdWidth; i++) {
      Product *curNode = (Product *)_vectors[it->second][i].get();

      Value *aggSerIn = ConstantFP::get(Type::getDoubleTy(_context), 1.0);

      for (auto &m : *curNode->multiplicands()) {
        Value *inVal;
        if (node2value.find(m->id()) == node2value.end()) {
          m->accept(*this, {});
        }
        if (node2value[m->id()].pos != -1) {
	  // input is in a vector
          if (_directVecInputs[it->second].find(node2value[m->id()].vec) !=
              _directVecInputs[it->second].end()) {
            // this input is part of a vector, which will be multiplied as a
            // whole
            continue;
          }
          inVal = _builder.CreateExtractElement(node2value[m->id()].val,
                                                node2value[m->id()].pos);
        } else {
          inVal = node2value[m->id()].val;
        }
        aggSerIn = _builder.CreateFMul(aggSerIn, inVal);
      }
      serialInputs.push_back(aggSerIn);
    }
    Value *out = _builder.CreateVectorSplat(simdWidth, serialInputs[0]);

    for (int i = 1; i < simdWidth; i++) {
      out = _builder.CreateInsertElement(out, serialInputs[i], i);
    }

    for (auto &m : *n.multiplicands()) {
      if (node2value[m->id()].pos == -1 ||
          _directVecInputs[it->second].find(node2value[m->id()].vec) ==
              _directVecInputs[it->second].end()) {
        // this input is not in a directly usable vector and already handled
        continue;
      }

      // TODO assert that lane order of input and output vec match

      out = _builder.CreateFMul(out, node2value[m->id()].val);
    }
    for (int i = 0; i < simdWidth; i++) {
      node2value.insert({_vectors[it->second][i]->id(), {out, i, it->second}});
    }
  }
}

void IREmitter::visitSum(Sum &n, arg_t arg) {
  // Node might have been handled already by another node it shares a vector
  // with
  if (node2value.find(n.id()) != node2value.end())
    return;
  auto it = _vec.find(n.id());
  if (it == _vec.end()) {
    Value *out = ConstantFP::get(Type::getDoubleTy(_context), 0.0);
    for (auto &m : *n.addends()) {
      if (node2value.find(m->id()) == node2value.end())
	m->accept(*this, {});
      auto in = node2value[m->id()];
      Value *inVal;
      if (in.pos == -1)
        inVal = in.val;
      else
        inVal = _builder.CreateExtractElement(in.val, in.pos);
      out = _builder.CreateFAdd(out, inVal);
    }
    node2value.insert({n.id(), {out, -1, 0}});
  } else {
    // operation is vectorized
    std::vector<Value *> serialInputs;
    // First emit all additions of inputs which do not come in a vector
    for (int i = 0; i < simdWidth; i++) {
      Sum *curNode = (Sum *)_vectors[it->second][i].get();

      Value *aggSerIn = ConstantFP::get(Type::getDoubleTy(_context), 0.0);

      for (auto &m : *curNode->addends()) {
	
        if (node2value.find(m->id()) == node2value.end()) {
          m->accept(*this, {});
        }
        Value *inVal;
        if (node2value[m->id()].pos != -1) {
	  // input is in a vector
          if (_directVecInputs[it->second].find(node2value[m->id()].vec) !=
              _directVecInputs[it->second].end()) {
            // this input is part of a vector, which will be multiplied as a
            // whole
            continue;
          }
          inVal = _builder.CreateExtractElement(node2value[m->id()].val,
                                                node2value[m->id()].pos);
        } else {
          inVal = node2value[m->id()].val;
        }
        aggSerIn = _builder.CreateFAdd(aggSerIn, inVal);
      }
      serialInputs.push_back(aggSerIn);
    }
    Value *out = _builder.CreateVectorSplat(simdWidth, serialInputs[0]);

    for (int i = 1; i < simdWidth; i++) {
      out = _builder.CreateInsertElement(out, serialInputs[i], i);
    }

    for (auto &m : *n.addends()) {
      if (node2value[m->id()].pos == -1 ||
          _directVecInputs[it->second].find(node2value[m->id()].vec) ==
              _directVecInputs[it->second].end()) {
        // this input is not in a directly usable vector and already handled
        continue;
      }

      // TODO assert that lane order of input and output vec match

      out = _builder.CreateFAdd(out, node2value[m->id()].val);
    }
    for (int i = 0; i < simdWidth; i++) {
      node2value.insert({_vectors[it->second][i]->id(), {out, i, it->second}});
    }
  }
}

void IREmitter::visitWeightedSum(WeightedSum &n, arg_t arg) {
  // Node might have been handled already by another node it shares a vector
  // with
  if (node2value.find(n.id()) != node2value.end())
    return;

  auto it = _vec.find(n.id());
  if (it == _vec.end()) {
    Value *out = ConstantFP::get(Type::getDoubleTy(_context), 0.0);
    for (auto &m : *n.addends()) {
      if (node2value.find(m.addend->id()) == node2value.end())
	m.addend->accept(*this, {});
      auto in = node2value[m.addend->id()];
      Value *inVal;
      if (in.pos == -1)
        inVal = in.val;
      else
        inVal = _builder.CreateExtractElement(in.val, in.pos);
      auto mul = _builder.CreateFMul(
          ConstantFP::get(Type::getDoubleTy(_context), m.weight), inVal);
      out = _builder.CreateFAdd(out, mul);
    }
    node2value.insert({n.id(), {out, -1, 0}});
  } else {
    // operation is vectorized
    std::vector<Value *> serialInputs;
    // First emit all additions of inputs which do not come in a vector
    for (int i = 0; i < simdWidth; i++) {
      WeightedSum *curNode = (WeightedSum *)_vectors[it->second][i].get();

      Value *aggSerIn = ConstantFP::get(Type::getDoubleTy(_context), 0.0);

      for (auto &m : *curNode->addends()) {
        if (node2value.find(m.addend->id()) == node2value.end()) {
          m.addend->accept(*this, {});
        }
        Value *inVal;
        if (node2value[m.addend->id()].pos != -1) {
	  // input is in a vector
          if (_directVecInputs[it->second].find(node2value[m.addend->id()].vec) !=
              _directVecInputs[it->second].end()) {
            // this input is part of a vector, which will be multiplied as a
            // whole
            continue;
          }
          inVal = _builder.CreateExtractElement(node2value[m.addend->id()].val,
                                                node2value[m.addend->id()].pos);
        } else {
          inVal = node2value[m.addend->id()].val;
        }
        auto mul = _builder.CreateFMul(
            ConstantFP::get(Type::getDoubleTy(_context), m.weight),
            inVal);
        aggSerIn = _builder.CreateFAdd(aggSerIn, mul);
      }
      serialInputs.push_back(aggSerIn);
    }
    Value *out = _builder.CreateVectorSplat(simdWidth, serialInputs[0]);

    for (int i = 1; i < simdWidth; i++) {
      out = _builder.CreateInsertElement(out, serialInputs[i], i);
    }

    for (auto &m : *n.addends()) {
      auto inIt = _vec.find(m.addend->id());
      if (node2value[m.addend->id()].pos == -1 ||
          _directVecInputs[it->second].find(node2value[m.addend->id()].vec) ==
              _directVecInputs[it->second].end()) {
        // this input is not in a directly usable vector and already handled
        continue;
      }
      // we now need to gather all weights for multiplication

      std::vector<Value *> weights;

      for (int i = 0; i < simdWidth; i++) {
        std::string inputName = _vectors[inIt->second][i]->id();
        auto inputs = ((WeightedSum *)_vectors[it->second][i].get())->addends();
        double weight;
        bool found = false;
        for (auto &w : *inputs) {
          if (w.addend->id() == inputName) {
            weight = w.weight;
            found = true;
          }
        }
        assert(found);
        weights.push_back(ConstantFP::get(Type::getDoubleTy(_context), weight));
      }

      Value *weightVec = _builder.CreateVectorSplat(simdWidth, weights[0]);

      for (int i = 1; i < simdWidth; i++) {
        weightVec = _builder.CreateInsertElement(weightVec, weights[i], i);
      }
      auto mul =
          _builder.CreateFMul(node2value[m.addend->id()].val, weightVec);
      out = _builder.CreateFAdd(out, mul);
    }
    for (int i = 0; i < simdWidth; i++) {
      node2value.insert({_vectors[it->second][i]->id(), {out, i, it->second}});
    }
  }
}

std::unordered_map<std::string, irVal>
IREmitter::getNodeMap() {
  return node2value;
}
