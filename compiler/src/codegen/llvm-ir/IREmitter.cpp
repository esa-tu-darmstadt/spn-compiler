#include "codegen/llvm-ir/IREmitter.h"

llvm::cl::opt<bool> useGather(
    "useGather",
    llvm::cl::desc("Use AVX2 Gather instructions to load histograms"),
    llvm::cl::cat(SPNCompiler));
llvm::cl::opt<bool> selectBinary(
    "selectBinary",
    llvm::cl::desc("Use select instructions instead of histograms loads for histograms with only two buckets"),
    llvm::cl::cat(SPNCompiler));
IREmitter::IREmitter(
      std::unordered_map<std::string, size_t> &vec,
      std::unordered_map<size_t, std::unordered_set<size_t>>& directVecInputs,
      std::vector<std::vector<NodeReference>>& vectors, Value *in,
      Function *func, LLVMContext &context, IRBuilder<> &builder,
      Module *module, unsigned width)
    : _vec(vec), _directVecInputs(directVecInputs), _vectors(vectors), _in(in),
      _func(func), _context(context), _builder(builder), _module(module),
      simdWidth(width) {
  for (auto& vec : directVecInputs) {
    for (auto& inVec : vec.second) {
      vecsWithOrder.insert(inVec);
    }
  }
  constantZero = ConstantFP::get(Type::getDoubleTy(_context), 0.0);
  constantOne = ConstantFP::get(Type::getDoubleTy(_context), 1.0);
}

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

Value* IREmitter::getHistogramPtr(Histogram& n) {
  std::vector<Constant *> values;
  size_t max_index = 0;
  for (auto &b : *n.buckets()) {
    for (int i = 0; i < (b.upperBound - b.lowerBound); ++i) {
      values.push_back(ConstantFP::get(Type::getDoubleTy(_context), b.value));
    }
    max_index = (b.upperBound > max_index) ? b.upperBound : max_index;
  }
  assert(max_index == values.size());
  auto arrayType = ArrayType::get(Type::getDoubleTy(_context), values.size());
  auto initializer = ConstantArray::get(arrayType, values);
  // XXX This leaks?
  auto arr = new GlobalVariable(*_module, arrayType, true,
                                GlobalValue::InternalLinkage, initializer, "histo" + n.id());
  return _builder.CreateGEP(arr,
                            {ConstantInt::get(Type::getInt64Ty(_context), 0),
                             ConstantInt::get(Type::getInt64Ty(_context), 0)});
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
  // TODO replace this with a sensible heuristic
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
    auto globalArray = getHistogramPtr(n);
    auto address = _builder.CreateGEP(
        globalArray, {in});
    auto out = _builder.CreateLoad(address);
    out->setName(n.id());
    node2value.insert({n.id(), {out, -1, 0}});
  }
}

void IREmitter::visitProduct(Product &n, arg_t arg) {
  emitArith(n, arg);
}

void IREmitter::visitSum(Sum &n, arg_t arg) {
  emitArith(n, arg);
}

void IREmitter::visitWeightedSum(WeightedSum &n, arg_t arg) {
  emitArith(n, arg);
}

std::unordered_map<std::string, irVal>
IREmitter::getNodeMap() {
  return node2value;
}
