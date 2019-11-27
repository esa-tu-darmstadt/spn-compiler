//
// Created by ls on 10/9/19.
//

#include "LLVMCodegen.h"
#include "transform/ExecOrderProducer.h"
#include "codegen/shared/VectorizationTraversal.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <queue>

#define SIMD_WIDTH 2
#define MIN_LENGTH 3

LLVMCodegen::LLVMCodegen() : builder{context} {
    module = std::make_unique<Module>("spn-llvm", context);
}

void LLVMCodegen::emitInput(InputVar *n, Value *in) {
  auto addr = builder.CreateGEP(
      in, {ConstantInt::getSigned(Type::getInt64Ty(context), n->index())});
  auto val = builder.CreateLoad(Type::getInt32Ty(context), addr);
  input2value.insert({n->id(), val});
}

void LLVMCodegen::emitHistogram(Histogram *n) {
  auto in = input2value[n->indexVar()->id()];
  auto& buckets = *n->buckets();
  auto inBlock = builder.GetInsertBlock();
  auto exit = BasicBlock::Create(context, "exit" + n->id(), func);
  builder.SetInsertPoint(exit);
  auto out = builder.CreatePHI(Type::getDoubleTy(context), buckets.size());
  node2value.insert({n->id(), out});
  builder.SetInsertPoint(inBlock);
  std::vector<BasicBlock*> matches;
  for (auto& b : buckets) {
    auto ge = builder.CreateICmpSGE(
        in, ConstantInt::getSigned(Type::getInt32Ty(context), b.lowerBound));
    auto lt = builder.CreateICmpSLT(
        in, ConstantInt::getSigned(Type::getInt32Ty(context), b.upperBound));
    auto cmp = builder.CreateAnd(ge, lt);
    auto inRange = BasicBlock::Create(
        context, "matched " + n->id() + ": " +
	std::to_string(b.lowerBound) + "-" + std::to_string(b.upperBound), func);
    matches.push_back(inRange);
    auto nextIf = BasicBlock::Create(context, "nextIf " + n->id() + ": " +
	std::to_string(b.lowerBound) + "-" + std::to_string(b.upperBound), func);
    builder.CreateCondBr(cmp, inRange, nextIf);
    builder.SetInsertPoint(inRange);
    builder.CreateBr(exit);
    out->addIncoming(ConstantFP::get(Type::getDoubleTy(context), b.value), inRange);
    builder.SetInsertPoint(nextIf);
  }
  
  // The input var is outside the range of the histogram, so we abort
  auto exitFuncType =
      FunctionType::get(Type::getVoidTy(context), Type::getInt32Ty(context), true);
  auto exitFunc = module->getOrInsertFunction("exit", exitFuncType);
  builder.CreateCall(exitFunc, ConstantInt::get(Type::getInt32Ty(context),1));
  builder.CreateRetVoid();
  builder.SetInsertPoint(exit);
}
void LLVMCodegen::emitProduct(Product *n) {
  Value* out = ConstantFP::get(Type::getDoubleTy(context), 1.0);
  for (auto& m : *n->multiplicands()) {
    auto in = node2value[m->id()];
    out = builder.CreateFMul(out, in);
  }
  node2value.insert({n->id(), out});
}
void LLVMCodegen::emitSum(Sum *n) {
  Value* out = ConstantFP::get(Type::getDoubleTy(context), 0.0);
  for (auto& a : *n->addends()) {
    out = builder.CreateFAdd(out, node2value[a->id()]);
  }
  node2value.insert({n->id(), out});
}
void LLVMCodegen::emitWeightedSum(WeightedSum *n) {
  Value* out = ConstantFP::get(Type::getDoubleTy(context), 0.0);
  for (auto& a : *n->addends()) {
    auto mul =
        builder.CreateFMul(ConstantFP::get(Type::getDoubleTy(context), a.weight),
                          node2value[a.addend->id()]);
    out = builder.CreateFAdd(out, mul);
  }
  node2value.insert({n->id(), out});
}

void LLVMCodegen::emitStore(GraphIRNode* n, Value* addr) {
  auto val = node2value[n->id()];
  builder.CreateStore(val, addr);
}


void LLVMCodegen::emitVecBody(IRGraph &graph, Value* in, Value* out) {
  // Perform BFS to find starting tree level
  std::vector<NodeReference> vectorRoots;
  class bfsBuilder : public BaseVisitor {
  public:
    void visitInputvar(InputVar &n, arg_t arg) {}
    void visitHistogram(Histogram& n, arg_t arg){
      q.push({currentLevel + 1, n.indexVar()});
    }
    void visitProduct(Product &n, arg_t arg) {
      for (auto &c : *n.multiplicands()) {
        q.push({currentLevel + 1, c});
      }
    }

    void visitSum(Sum &n, arg_t arg) {
      for (auto &c : *n.addends()) {
        q.push({currentLevel + 1, c});
      }
    }

    void visitWeightedSum(WeightedSum &n, arg_t arg) {
      for (auto &c : *n.addends()) {
        q.push({currentLevel + 1, c.addend});
      }
    }
    std::queue<std::pair<size_t, NodeReference>> q;
    size_t currentLevel = 0;
  };

  bfsBuilder visitor;
  graph.rootNode->accept(visitor, {});
  while (!visitor.q.empty()) {
    if (visitor.currentLevel < visitor.q.front().first) {
      if (vectorRoots.size() >= SIMD_WIDTH) {
        break;
      } else {
        visitor.currentLevel++;
	vectorRoots.clear();
      }
    }
    visitor.q.front().second->accept(visitor, {});
    vectorRoots.push_back(visitor.q.front().second);
    visitor.q.pop();
  }

  // TODO if no rootSet found, go back to emitting serial
  
  std::queue<std::pair<std::vector<NodeReference>, std::unordered_set<std::string>>> rootSetQueue;
  
  rootSetQueue.push({vectorRoots, {}});

  std::vector<std::vector<std::vector<NodeReference>>> sequences;
  while (!rootSetQueue.empty()) {
    auto roots = rootSetQueue.front().first;
    std::unordered_set<std::string> pruned = rootSetQueue.front().second;
    auto nodeGroupSequence = getLongestChain(roots, pruned);

    while (nodeGroupSequence.size() > 0) {
      sequences.push_back(nodeGroupSequence);
      std::vector<NodeReference> newRoots;
      for (auto &n : nodeGroupSequence[1]) {
        pruned.insert(n->id());
	newRoots.push_back(n);
      }
      std::unordered_set<std::string> newPrunes;
      for (auto &n : nodeGroupSequence[2]) {
	newPrunes.insert(n->id());
      }
      rootSetQueue.push({newRoots, newPrunes});

      std::cout << "new chain " << std::endl;
      for (auto &e : nodeGroupSequence) {
        std::cout << "new ins " << std::endl;
        for (auto &n : e)
          std::cout << "id " << n->id() << std::endl;
      }
      
      nodeGroupSequence = getLongestChain(roots, pruned);

    }
    rootSetQueue.pop();
  }

  std::unordered_map<std::string, std::vector<NodeReference>> vectedNodes;

  for (auto& seq : sequences) {
    for (auto& instr : seq) {
      for (auto& lane : instr) {
	vectedNodes.insert({lane->id(), instr});
      }
    }
  }

  std::stack<NodeReference> instrStack;
  instrStack.push(graph.rootNode);
  
  bfsBuilder stackBuilder;
  graph.rootNode->accept(stackBuilder, {});
  while (!stackBuilder.q.empty()) {
    auto& ref = stackBuilder.q.front();
    instrStack.push(ref.second);
    ref.second->accept(stackBuilder, {});
    stackBuilder.q.pop();
  }

  class vecCodeGen : public BaseVisitor {
  public:
    vecCodeGen(std::unordered_map<std::string, std::vector<NodeReference>> vec,
               Value *in, Function *func, LLVMContext &context,
               IRBuilder<> &builder, Module* module)
        : _vec(vec), _in(in), _func(func), _context(context),
          _builder(builder), _module(module) {}
    
    void visitInputvar(InputVar &n, arg_t arg) {
      // Not vectorized at the moment
      auto addr = _builder.CreateGEP(
          _in, {ConstantInt::getSigned(Type::getInt64Ty(_context), n.index())});
      auto val = _builder.CreateLoad(Type::getInt32Ty(_context), addr);
      input2value.insert({n.id(), val});
    }
    
    void visitHistogram(Histogram &n, arg_t arg) {
      // TODO Vectorized version generated, not emitted atm, use avx gather and
      // constantarray (when sensible)
      auto in = input2value[n.indexVar()->id()];
      auto &buckets = *n.buckets();
      auto inBlock = _builder.GetInsertBlock();
      auto exit = BasicBlock::Create(_context, "exit" + n.id(), _func);
      _builder.SetInsertPoint(exit);
      auto out = _builder.CreatePHI(Type::getDoubleTy(_context), buckets.size());
      node2value.insert({n.id(), {out, -1}});
      _builder.SetInsertPoint(inBlock);
      std::vector<BasicBlock *> matches;
      for (auto &b : buckets) {
        auto ge = _builder.CreateICmpSGE(
            in,
            ConstantInt::getSigned(Type::getInt32Ty(_context), b.lowerBound));
        auto lt = _builder.CreateICmpSLT(
            in,
            ConstantInt::getSigned(Type::getInt32Ty(_context), b.upperBound));
        auto cmp = _builder.CreateAnd(ge, lt);
        auto inRange = BasicBlock::Create(
            _context,
            "matched " + n.id() + ": " + std::to_string(b.lowerBound) + "-" +
                std::to_string(b.upperBound),
            _func);
        matches.push_back(inRange);
        auto nextIf = BasicBlock::Create(_context,
                                         "nextIf " + n.id() + ": " +
                                             std::to_string(b.lowerBound) +
                                             "-" + std::to_string(b.upperBound),
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
    }
    
    void visitProduct(Product &n, arg_t arg) {
      // Node might have been handled already by another node it shares a vector with
      if (node2value.find(n.id()) != node2value.end())
	return;
      auto it = _vec.find(n.id());
      if (it == _vec.end()) {
        Value *out = ConstantFP::get(Type::getDoubleTy(_context), 1.0);
        for (auto &m : *n.multiplicands()) {
          auto in = node2value[m->id()];
	  Value* inVal;
	  if (in.second == -1)
	    inVal = in.first;
	  else
	    inVal = _builder.CreateExtractElement(in.first, in.second);
          out = _builder.CreateFMul(out, inVal);
        }
        node2value.insert({n.id(), {out, -1}});
      } else {
	// operation is vectorized
	std::array<Value*, SIMD_WIDTH> serialInputs;
	// First emit all multiplications of inputs which do not come in a vector
        for (int i = 0; i < SIMD_WIDTH; i++) {
	  Product* curNode = (Product*) it->second[i].get();
	  
	  Value *aggSerIn = ConstantFP::get(Type::getDoubleTy(_context), 1.0);
	  
	  for (auto& m : *curNode->multiplicands()) {
	    if (node2value[m->id()].second != -1) {
	      // this input is part of a vector, which will be multiplied as a whole
	      continue;
	    }
	    aggSerIn = _builder.CreateFMul(aggSerIn, node2value[m->id()].first);
	  }
	  serialInputs[i] = aggSerIn;
        }
	Value* out = _builder.CreateVectorSplat(SIMD_WIDTH, serialInputs[0]);

	for (int i = 1; i < SIMD_WIDTH; i++) {
	  out = _builder.CreateInsertElement(out, serialInputs[i], i);
	}

        for (auto &m : *n.multiplicands()) {
          if (node2value[m->id()].second == -1) {
            // this input is not in a vector and already handled
            continue;
          }

          out = _builder.CreateFMul(out, node2value[m->id()].first);
        }
	for (int i  = 0; i < SIMD_WIDTH; i++) {
	  node2value.insert({it->second[i]->id(), {out, i}});
	}
	
      }
    }

    void visitSum(Sum &n, arg_t arg) {
      // Node might have been handled already by another node it shares a vector with
      if (node2value.find(n.id()) != node2value.end())
	return;
      auto it = _vec.find(n.id());
      if (it == _vec.end()) {
        Value *out = ConstantFP::get(Type::getDoubleTy(_context), 0.0);
        for (auto &m : *n.addends()) {
          auto in = node2value[m->id()];
	  Value* inVal;
	  if (in.second == -1)
	    inVal = in.first;
	  else
	    inVal = _builder.CreateExtractElement(in.first, in.second);
          out = _builder.CreateFAdd(out, inVal);
        }
        node2value.insert({n.id(), {out, -1}});
      } else {
	// operation is vectorized
	std::array<Value*, SIMD_WIDTH> serialInputs;
	// First emit all additions of inputs which do not come in a vector
        for (int i = 0; i < SIMD_WIDTH; i++) {
	  Sum* curNode = (Sum*) it->second[i].get();
	  
	  Value *aggSerIn = ConstantFP::get(Type::getDoubleTy(_context), 0.0);
	  
	  for (auto& m : *curNode->addends()) {
	    if (node2value[m->id()].second != -1) {
	      // this input is part of a vector, which will be added as a whole
	      continue;
	    }
	    
	    aggSerIn = _builder.CreateFAdd(aggSerIn, node2value[m->id()].first);
	  }
	  serialInputs[i] = aggSerIn;
        }
	Value* out = _builder.CreateVectorSplat(SIMD_WIDTH, serialInputs[0]);

	for (int i = 1; i < SIMD_WIDTH; i++) {
	  out = _builder.CreateInsertElement(out, serialInputs[i], i);
	}

        for (auto &m : *n.addends()) {
          if (node2value[m->id()].second == -1) {
            // this input is not in a vector and already handled
            continue;
          }

          out = _builder.CreateFAdd(out, node2value[m->id()].first);
        }
	for (int i  = 0; i < SIMD_WIDTH; i++) {
	  node2value.insert({it->second[i]->id(), {out, i}});
	}
	
      }
    }

    void visitWeightedSum(WeightedSum &n, arg_t arg) {
      // Node might have been handled already by another node it shares a vector with
      if (node2value.find(n.id()) != node2value.end())
	return;
      
      auto it = _vec.find(n.id());
      if (it == _vec.end()) {
        Value *out = ConstantFP::get(Type::getDoubleTy(_context), 0.0);
        for (auto &m : *n.addends()) {
          auto in = node2value[m.addend->id()];
	  Value* inVal;
	  if (in.second == -1)
	    inVal = in.first;
	  else
	    inVal = _builder.CreateExtractElement(in.first, in.second);
          auto mul = _builder.CreateFMul(
              ConstantFP::get(Type::getDoubleTy(_context), m.weight), inVal);
          out = _builder.CreateFAdd(out, mul);
        }
        node2value.insert({n.id(), {out, -1}});
      } else {
	// operation is vectorized
	std::array<Value*, SIMD_WIDTH> serialInputs;
	// First emit all additions of inputs which do not come in a vector
        for (int i = 0; i < SIMD_WIDTH; i++) {
	  WeightedSum* curNode = (WeightedSum*) it->second[i].get();
	  
	  Value *aggSerIn = ConstantFP::get(Type::getDoubleTy(_context), 0.0);
	  
	  for (auto& m : *curNode->addends()) {
	    if (node2value[m.addend->id()].second != -1) {
	      // this input is part of a vector, which will be added as a whole
	      continue;
	    }
	    auto mul = _builder.CreateFMul(
              ConstantFP::get(Type::getDoubleTy(_context), m.weight), node2value[m.addend->id()].first);
            aggSerIn = _builder.CreateFAdd(aggSerIn, mul);
          }
	  serialInputs[i] = aggSerIn;
        }
	Value* out = _builder.CreateVectorSplat(SIMD_WIDTH, serialInputs[0]);

	for (int i = 1; i < SIMD_WIDTH; i++) {
	  out = _builder.CreateInsertElement(out, serialInputs[i], i);
	}

        for (auto &m : *n.addends()) {
	  auto inIt = _vec.find(m.addend->id());
          if (node2value[m.addend->id()].second == -1) {
            // this input is not in a vector and has already been handled
            continue;
          }
	  // we now need to gather all weights for multiplication

          std::array<Value *, SIMD_WIDTH> weights;

	  for (int i = 0; i < SIMD_WIDTH; i++) {
	    std::string inputName = inIt->second[i]->id();
	    auto inputs = ((WeightedSum*) it->second[i].get())->addends();
	    double weight;
	    bool found = false;
	    for (auto& w : *inputs) {
	      if (w.addend->id() == inputName) {
		weight = w.weight;
		found = true;
	      }
	    }
	    assert(found);
	    weights[i] = ConstantFP::get(Type::getDoubleTy(_context), weight);
	  }

          Value *weightVec = _builder.CreateVectorSplat(SIMD_WIDTH, weights[0]);

          for (int i = 1; i < SIMD_WIDTH; i++) {
            weightVec = _builder.CreateInsertElement(weightVec, weights[i], i);
          }
	  auto mul = _builder.CreateFMul(node2value[m.addend->id()].first, weightVec);
          out = _builder.CreateFAdd(out, mul);
        }
	for (int i  = 0; i < SIMD_WIDTH; i++) {
	  node2value.insert({it->second[i]->id(), {out, i}});
	}
      }
    }

    std::unordered_map<std::string, std::pair<Value *, int>> getNodeMap() {
      return node2value;
    }

  private:
    std::unordered_map<std::string, std::vector<NodeReference>> _vec;
    Value* _in;
    Function* _func;
    LLVMContext& _context;
    IRBuilder<>& _builder;
    Module* _module;
    std::unordered_map<std::string, std::pair<Value*, int>> node2value;
    std::unordered_map<std::string, Value *> input2value;
  };

  vecCodeGen codeEmitter(vectedNodes, in, func, context, builder, module.get());
  while (!instrStack.empty()) {
    instrStack.top()->accept(codeEmitter, {});
    instrStack.pop();
  }

  builder.CreateStore(codeEmitter.getNodeMap()[graph.rootNode->id()].first, out);
}

std::vector<std::vector<NodeReference>>
LLVMCodegen::getLongestChain(std::vector<NodeReference> vectorRoots,
                             std::unordered_set<std::string> pruned) {
  // lexicographically sorted, thus paths with longest shared prefixes are next
  // to each other
  std::multimap<std::string, std::pair<size_t, size_t>> paths;
  std::vector<std::vector<std::vector<NodeReference>>> raw_paths;
  auto trav = VectorizationTraversal(pruned);
  for (unsigned i = 0; i < vectorRoots.size(); i++) {
    auto res = trav.collectPaths(vectorRoots[i]);
    raw_paths.push_back({});
    for (int j = 0; j < res.size(); j++) {
      paths.insert({res[j].first, {i, j}});
      raw_paths[i].push_back(res[j].second);
    }
  }

  int maxPrefixLength = -1;
  std::vector<decltype(paths.rbegin())> maxPaths;

  /* Use reverse iterator so that for two paths starting at the same root, where
     one path's operation order is a prefix of the path's operation order, the
     longer path is considered
  */
  for (auto rIt = paths.rbegin(); std::distance(rIt, paths.rend()) >= SIMD_WIDTH; rIt++) {
    std::unordered_map<size_t, decltype(rIt)> independentPaths;
    independentPaths.insert({rIt->second.first, rIt});
    auto candidate = std::next(rIt);
    // Find SIMD_WIDTH independent paths that are lexicographically smaller than
    // rIt's path but lexicographically as close as possible to rIt's path
    while (independentPaths.size() < SIMD_WIDTH && candidate != paths.rend()) {
      if (independentPaths.find(candidate->second.first) ==
          independentPaths.end()) {
	// found the next smaller path to rIt's for a root, that has not been selected yet
	independentPaths.insert({candidate->second.first, candidate});
      }
      candidate++;
    }

    if (independentPaths.size() < SIMD_WIDTH) {
      continue;
    }

    // Now check, how long the common prefix of the paths in independentPaths is

    auto it = independentPaths.begin();
    std::string prefix = it->second->first;

    for (it = std::next(it); it != independentPaths.end(); it++) {
      std::string newPrefix;
      int preLength = prefix.length(), pathLength = it->second->first.length();
      for (int i = 0; i < preLength && i < pathLength; i++) {
        if (prefix[i] != it->second->first[i])
          break;
        newPrefix.push_back(prefix[i]);
      }
      prefix = newPrefix;
    }
    int preLen = prefix.length();
    if (preLen > maxPrefixLength) {
      // Found a longer isomorphic chain
      maxPrefixLength = prefix.length();
      std::vector<decltype(paths.rbegin())> newPaths;
      for (auto& e : independentPaths) {
	newPaths.push_back(e.second);
      }
      maxPaths = newPaths;
    }
    
  }

  if (maxPrefixLength < MIN_LENGTH) {
    return {};
  }

  // TODO: Make the inner an array
  std::vector<std::vector<NodeReference>> nodeGroupSequence(maxPrefixLength);
  
  for (int i = 0; i < SIMD_WIDTH; i++) {
    auto &nodeVec =
        raw_paths[maxPaths[i]->second.first][maxPaths[i]->second.second];
    for (int j = 0; j < maxPrefixLength; j++) {
      nodeGroupSequence[j].push_back(nodeVec[j]);
    }
  }

  return nodeGroupSequence;
}

void LLVMCodegen::emitBody(IRGraph &graph, Value* in, Value* out) {

  ExecOrderProducer prod;
  prod.produceOrder(graph.rootNode);

  auto& nodes = prod.ordered_nodes();

  while (!nodes.empty()) {
    auto node = nodes.top();
    switch (node.kind) {
    case NodeKind::Input:
      emitInput((InputVar *)node.node, in);
      break;
    case NodeKind::Histogram:
      emitHistogram((Histogram *)node.node);
      break;
    case NodeKind::Product:
      emitProduct((Product *)node.node);
      break;
    case NodeKind::WeightedSum:
      emitWeightedSum((WeightedSum *)node.node);
      break;
    case NodeKind::Sum:
      emitSum((Sum *)node.node);
      break;
    case NodeKind::Store:
      emitStore(node.node, out);
      break;

    default:
      throw std::runtime_error("Unhandled NodeKind");
    }
    nodes.pop();
    // Storing the result should be the last operation
    assert(node.kind != NodeKind::Store || nodes.empty());
  }
}

void LLVMCodegen::generateLLVMIR(IRGraph &graph, bool vectorize) {
    auto intType = Type::getInt32Ty(context);
    //auto structElements = std::vector<Type*>(graph.inputs->size(), intType);
    //auto activationType = StructType::create(context, structElements, "activation_t", false);
    //auto activationPtrType = PointerType::get(activationType, 0);
    std::vector<Type*> argTypes{PointerType::get(intType, 0), Type::getDoublePtrTy(context, 0), Type::getInt64Ty(context)};
    auto functionType = FunctionType::get(Type::getVoidTy(context), argTypes, false);
    func = Function::Create(functionType, Function::ExternalLinkage, "spn_element", module.get());
    auto bb = BasicBlock::Create(context, "main", func);
    auto lh = BasicBlock::Create(context, "loop-header", func);
    auto lb = BasicBlock::Create(context, "loop-body", func);
    auto ex = BasicBlock::Create(context, "exit", func);
    
    auto arg_it = func->arg_begin();
    auto input = arg_it++;
    auto output = arg_it++;
    auto count = arg_it;

    builder.SetInsertPoint(bb);
    builder.CreateBr(lh);

    builder.SetInsertPoint(lh);
    auto cur_count = builder.CreatePHI(Type::getInt64Ty(context), 2);
    cur_count->addIncoming(ConstantInt::getSigned(Type::getInt64Ty(context), 0), bb);
    auto cmp = builder.CreateICmpSLT(cur_count, count);
    builder.CreateCondBr(cmp, lb, ex);

    builder.SetInsertPoint(lb);
    auto in_offset = builder.CreateMul(cur_count, ConstantInt::getSigned(Type::getInt64Ty(context), graph.inputs->size()));
    auto in_ptr = builder.CreateGEP(input, {in_offset});
    auto out_ptr = builder.CreateGEP(output, {cur_count});
    if (vectorize)
      emitVecBody(graph, in_ptr, out_ptr);
    else
      emitBody(graph, in_ptr, out_ptr);

    auto inc = builder.CreateAdd(cur_count, ConstantInt::getSigned(Type::getInt64Ty(context), 1));
    cur_count->addIncoming(inc, builder.GetInsertBlock());
    builder.CreateBr(lh);

    builder.SetInsertPoint(ex);
    builder.CreateRetVoid();
    
    std::error_code EC;
    llvm::raw_fd_ostream OS("/Users/johannesschulte/Desktop/Uni/MT/cpp-spn-compiler/debLLVMBuild/out.bc", EC);
    //module->print(llvm::errs(), nullptr);
    WriteBitcodeToFile(*module, OS);
}

