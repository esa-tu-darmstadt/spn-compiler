//
// Created by ls on 10/9/19.
//

#include "LLVMCodegen.h"
#include "codegen/shared/VectorizationTraversal.h"
#include "codegen/shared/BFSOrderProducer.h"
#include "codegen/llvm-ir/IREmitter.h"
#include "codegen/shared/PackingSolver.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <queue>

#define SIMD_WIDTH 2
#define MIN_LENGTH 2
#define USE_SOLVER 1

LLVMCodegen::LLVMCodegen() : builder{context} {
    module = std::make_unique<Module>("spn-llvm", context);
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

std::unordered_map<std::string, std::vector<NodeReference>> LLVMCodegen::getVectorization(IRGraph& graph) {
  // Perform BFS to find starting tree level
  std::vector<NodeReference> vectorRoots;
  BFSOrderProducer visitor;
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
      if (roots.size() > SIMD_WIDTH) {
	// Prevent other groupings than the one in nodeGroupSequence[0] by splitting the roots set
        roots.erase(std::remove_if(roots.begin(), roots.end(),
                                   [&](auto &nr) {
                                     for (auto &n : nodeGroupSequence[0]) {
                                       if (nr->id() == n->id()) {
                                         return true;
                                       }
                                     }
                                     return false;
                                   }),
                    roots.end());

        std::unordered_set<std::string> prunesOfSplit;
	for (auto &n : nodeGroupSequence[1]) {
          prunesOfSplit.insert(n->id());
        }
	rootSetQueue.push({nodeGroupSequence[0], prunesOfSplit});
      } else {
        for (auto &n : nodeGroupSequence[1]) {
          pruned.insert(n->id());
        }
      }

      // add new root sets along the chain
      for (int i = 1; i + 1 < nodeGroupSequence.size(); i++) {
        std::vector<NodeReference> newRoots;
        for (auto &n : nodeGroupSequence[i]) {
          newRoots.push_back(n);
        }
        std::unordered_set<std::string> newPrunes;
        for (auto &n : nodeGroupSequence[i+1]) {
          newPrunes.insert(n->id());
        }
        rootSetQueue.push({newRoots, newPrunes});
      }

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

  return vectedNodes;
}

void LLVMCodegen::generateLLVMIR(IRGraph &graph, bool vectorize) {
    auto intType = Type::getInt32Ty(context);
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

    std::unordered_map<std::string, size_t> partOf;
    std::unordered_map<size_t, std::unordered_set<size_t>> directVecInputs;
    std::unordered_map<size_t, std::vector<NodeReference>> vectors;
    if (vectorize) {
      if (USE_SOLVER) {
        PackingSolver solver;
        partOf = solver.getPacking(graph, SIMD_WIDTH);
	directVecInputs = solver.directVecInputs;
	vectors = solver.vectors;
      } else {
        // vecs = getVectorization(graph);
      }
    }

    IREmitter codeEmitter(partOf, directVecInputs, vectors, in_ptr, func,
                          context, builder, module.get(), SIMD_WIDTH);

    
    graph.rootNode->accept(codeEmitter, {});
    builder.CreateStore(codeEmitter.getNodeMap()[graph.rootNode->id()].val,
                        out_ptr);
    
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
