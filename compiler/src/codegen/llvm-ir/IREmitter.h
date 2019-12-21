#include <unordered_set>
#include <graph-ir/GraphIRNode.h>
#include "llvm/IR/IRBuilder.h"
#include "transform/BaseVisitor.h"
#include <unordered_map>

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
      std::unordered_map<size_t, std::unordered_set<size_t>>& directVecInputs,
      std::unordered_map<size_t, std::vector<NodeReference>>& vectors, Value *in,
      Function *func, LLVMContext &context, IRBuilder<> &builder,
      Module *module, unsigned width);

  void visitInputvar(InputVar &n, arg_t arg) override;

  void visitHistogram(Histogram &n, arg_t arg) override;

  void visitProduct(Product &n, arg_t arg) override;
  void visitSum(Sum &n, arg_t arg) override;
  void visitWeightedSum(WeightedSum &n, arg_t arg) override;

  std::unordered_map<std::string, irVal> getNodeMap();

private:
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

  Value *scalarArith(WeightedSum &n, Value *acc, Value *in, int inPos) {
    auto mul = _builder.CreateFMul(
        ConstantFP::get(Type::getDoubleTy(_context), (*n.addends())[inPos].weight), in);
    return _builder.CreateFAdd(acc, mul);
  }
  Value *scalarArith(Sum &n, Value *acc, Value *in, int inPos) {
    return _builder.CreateFAdd(acc, in);
  }
  
  Value *scalarArith(Product &n, Value *acc, Value *in, int inPos) {
    return _builder.CreateFMul(acc, in);
  }

  Value *vecArith(WeightedSum &n, Value *acc, int nPos) {
    auto m = getInput(n, nPos);
    auto inIt = _vec.find(m->id());
    std::vector<Value *> weights;

    for (int i = 0; i < simdWidth; i++) {
      std::string inputName = _vectors[inIt->second][i]->id();
      auto inputs = ((WeightedSum *)_vectors[_vec[n.id()]][i].get())->addends();
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
    auto mul = _builder.CreateFMul(node2value[m->id()].val, weightVec);
    return _builder.CreateFAdd(acc, mul);
  }

  Value *vecArith(Sum &n, Value *acc, int nPos) {
    return _builder.CreateFAdd(acc, node2value[getInput(n, nPos)->id()].val);
  }

  Value *vecArith(Product &n, Value *acc, int nPos) {
    return _builder.CreateFMul(acc, node2value[getInput(n, nPos)->id()].val);
  }
  
  template <class T> void emitArith(T &n) {
    // Node might have been handled already by another node it shares a vector
    // with
    if (node2value.find(n.id()) != node2value.end())
      return;

    auto it = _vec.find(n.id());
    if (it == _vec.end()) {
      Value *out = ConstantFP::get(Type::getDoubleTy(_context), getNeutralValue<T>());
      for (int i = 0; i < getInputLength(n); i++) {
        if (node2value.find(getInput(n,i)->id()) == node2value.end())
          getInput(n,i)->accept(*this, {});
        auto in = node2value[getInput(n,i)->id()];
        Value *inVal;
        if (in.pos == -1)
          inVal = in.val;
        else
          inVal = _builder.CreateExtractElement(in.val, in.pos);
	out = scalarArith(n, out, inVal, i);
      }
      node2value.insert({n.id(), {out, -1, 0}});
    } else {
      // operation is vectorized
      std::vector<Value *> serialInputs;
      // First emit all reductions of inputs which do not come in a vector
      for (int i = 0; i < simdWidth; i++) {
        T *curNode = (T *)_vectors[it->second][i].get();

        Value *aggSerIn = ConstantFP::get(Type::getDoubleTy(_context), getNeutralValue<T>());

        for (int j = 0; j < getInputLength(*curNode); j++) {
          NodeReference m = getInput(*curNode, j);
          if (node2value.find(m->id()) == node2value.end()) {
            m->accept(*this, {});
          }
          Value *inVal;
          if (node2value[m->id()].pos != -1) {
            // input is in a vector
            if (_directVecInputs[it->second].find(node2value[m->id()].vec) !=
                _directVecInputs[it->second].end()) {
              // this input is part of a vector, which will be reduced as a
              // whole
              continue;
            }
            inVal = _builder.CreateExtractElement(node2value[m->id()].val,
                                                  node2value[m->id()].pos);
          } else {
            inVal = node2value[m->id()].val;
          }

          aggSerIn = scalarArith(*curNode, aggSerIn, inVal, j);
        }
        serialInputs.push_back(aggSerIn);
      }
      Value *out = _builder.CreateVectorSplat(simdWidth, serialInputs[0]);

      for (int i = 1; i < simdWidth; i++) {
        out = _builder.CreateInsertElement(out, serialInputs[i], i);
      }

      for (int j = 0; j < getInputLength(n); j++) {
        NodeReference m = getInput(n, j);
        if (node2value[m->id()].pos == -1 ||
            _directVecInputs[it->second].find(node2value[m->id()].vec) ==
                _directVecInputs[it->second].end()) {
          // this input is not in a directly usable vector and already handled
          continue;
        }
	out = vecArith(n, out, j);
      }
      for (int i = 0; i < simdWidth; i++) {
        node2value.insert(
            {_vectors[it->second][i]->id(), {out, i, it->second}});
      }
    }
  }
  std::unordered_map<std::string, size_t>& _vec;
  std::unordered_map<size_t, std::unordered_set<size_t>>& _directVecInputs;
  std::unordered_map<size_t, std::vector<NodeReference>>& _vectors;
  Value *_in;
  Function *_func;
  LLVMContext &_context;
  IRBuilder<> &_builder;
  Module *_module;
  std::unordered_map<std::string, irVal> node2value;
  std::unordered_map<std::string, Value *> input2value;
  unsigned simdWidth;
};
