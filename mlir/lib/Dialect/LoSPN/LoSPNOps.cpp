//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPN/LoSPNOps.h"
#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNAttributes.h"
#include "LoSPN/LoSPNInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
  namespace spn {
    namespace low {

      static mlir::LogicalResult verifyKernel(SPNKernel kernel) {
        // Check that for each input of the SPNKernel the entry block of the Kernel has
        // a block argument with identical type.
        if (kernel.body().front().getNumArguments() != kernel.getType().getNumInputs()) {
          return kernel.emitOpError() << "Number of entry block arguments does not match number of Kernel inputs";
        }
        for (auto inputAndBlockArg : llvm::zip(kernel.getType().getInputs(), kernel.body().front().getArguments())) {
          if (std::get<0>(inputAndBlockArg) != std::get<1>(inputAndBlockArg).getType()) {
            return kernel.emitOpError() << "Kernel input type " << std::get<0>(inputAndBlockArg)
                                        << " and block argument type " << std::get<1>(inputAndBlockArg).getType()
                                        << " did not match";
          }
        }

        // Check that the values returned by each return found directly inside the Kernel
        // match the output type of the Kernel.
        if (!kernel.getType().getResults().empty()) {
          for (auto ret : kernel.body().getOps<SPNReturn>()) {
            if (ret.returnValues().size() != kernel.getType().getNumResults()) {
              return kernel.emitOpError() << "Number of return values does not match Kernel result type";
            }
            for (auto typeAndValue : llvm::zip(kernel.getType().getResults(), ret.returnValues())) {
              auto retTy = std::get<0>(typeAndValue);
              auto valTy = std::get<1>(typeAndValue).getType();
              if (retTy != valTy) {
                return kernel.emitOpError() << "Kernel result type " << retTy
                                            << " did not match return value type " << valTy;
              }
            }
          }
        }
        return mlir::success();
      }

      static mlir::LogicalResult verifyTask(SPNTask task) {
        // Check that the first argument of the entry block
        // has IndexType (corresponds to the batch index) and
        // the remaining arguments match the types of the
        // Tasks' operands.
        if (task.body().front().getNumArguments() != task.getNumOperands() + 1) {
          return task->emitOpError() << "Incorrect number of block arguments for entry block of Task";
        }
        for (auto blockArg : llvm::enumerate(task.body().front().getArguments())) {
          if (blockArg.index() == 0) {
            if (!blockArg.value().getType().isIndex()) {
              return task.emitOpError() << "First argument of Task block must be an index";
            }
          } else {
            if (blockArg.value().getType() != task->getOperand(blockArg.index() - 1).getType()) {
              return task.emitOpError() << "Task block argument type does not match Task operand type";
            }
          }
        }
        // Check that the task is terminated by a SPNReturn with the correct number of return values
        // and types.
        auto ret = dyn_cast<SPNReturn>(task.body().front().getTerminator());
        assert(ret);
        if (ret.returnValues().size() != task.results().size()) {
          return task.emitOpError() << "Task does not return the correct number of values";
        }
        for (auto retVal : llvm::zip(ret.returnValues(), task.results())) {
          if (std::get<0>(retVal).getType() != std::get<1>(retVal).getType()) {
            return task->emitOpError() << "Returned value type does not match Task result type";
          }
        }
        return mlir::success();
      }

    }
  }
}

//===----------------------------------------------------------------------===//
// SPNKernel
//===----------------------------------------------------------------------===//

void mlir::spn::low::SPNKernel::build(::mlir::OpBuilder& odsBuilder,
                                      ::mlir::OperationState& odsState,
                                      StringRef name,
                                      FunctionType type) {
  auto nameAttr = odsBuilder.getStringAttr(name);
  auto typeAttr = TypeAttr::get(type);
  odsState.addAttribute(SymbolTable::getSymbolAttrName(), nameAttr);
  odsState.addAttribute(getTypeAttrName(), typeAttr);
  odsState.addRegion();
}

//===----------------------------------------------------------------------===//
// SPNTask
//===----------------------------------------------------------------------===//

mlir::Block* mlir::spn::low::SPNTask::addEntryBlock() {
  assert(body().empty() && "Task already has a block");
  auto* entry = new Block();
  body().push_back(entry);
  entry->addArgument(IndexType::get(this->getContext()));
  entry->addArguments(this->inputs().getType());
  return entry;
}

mlir::Value mlir::spn::low::SPNTask::getBatchIndex() {
  assert(!body().empty() && "Task has no block");
  assert(body().front().getNumArguments() >= 1 && "Task block has no argument");
  return body().front().getArgument(0);
}

//===----------------------------------------------------------------------===//
// SPNConstant
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult mlir::spn::low::SPNConstant::fold(::llvm::ArrayRef<::mlir::Attribute> operands) {
  assert(operands.empty() && "lo_spn.constant has no operands");
  return valueAttr();
}

//===----------------------------------------------------------------------===//
// SPNGaussianLeaf
//===----------------------------------------------------------------------===//

bool mlir::spn::low::SPNGaussianLeaf::isVectorizable(unsigned vectorFactor) {
  // Floating point narrowing (FPTrunc) and widening (FPExt) cannot be performed in
  // vectorized mode, hence vectorization is not possible if such a transformation
  // of the input value is required.
  if (auto inputFloatType = this->index().getType().dyn_cast<FloatType>()) {
    if (auto outputFloatType = this->getResult().getType().dyn_cast<FloatType>()) {
      if (inputFloatType.getWidth() != outputFloatType.getWidth()) {
        return false;
      }
    }
  }
  return true;
}

#define GET_OP_CLASSES
#include "LoSPN/LoSPNOps.cpp.inc"