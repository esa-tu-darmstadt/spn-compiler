//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

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
        if (kernel.getBody().front().getNumArguments() != kernel.getFunctionType().getNumInputs()) {
          return kernel.emitOpError() << "Number of entry block arguments does not match number of Kernel inputs";
        }
        for (auto inputAndBlockArg : llvm::zip(kernel.getFunctionType().getInputs(), kernel.getBody().front().getArguments())) {
          if (std::get<0>(inputAndBlockArg) != std::get<1>(inputAndBlockArg).getType()) {
            return kernel.emitOpError() << "Kernel input type " << std::get<0>(inputAndBlockArg)
                                        << " and block argument type " << std::get<1>(inputAndBlockArg).getType()
                                        << " did not match";
          }
        }

        // Check that the values returned by each return found directly inside the Kernel
        // match the output type of the Kernel.
        if (!kernel.getFunctionType().getResults().empty()) {
          for (auto ret : kernel.getBody().getOps<SPNReturn>()) {
            if (ret.getReturnValues().size() != kernel.getFunctionType().getNumResults()) {
              return kernel.emitOpError() << "Number of return values does not match Kernel result type";
            }
            for (auto typeAndValue : llvm::zip(kernel.getFunctionType().getResults(), ret.getReturnValues())) {
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
        if (task.getBody().front().getNumArguments() != task.getNumOperands() + 1) {
          return task->emitOpError() << "Incorrect number of block arguments for entry block of Task";
        }
        for (auto blockArg : llvm::enumerate(task.getBody().front().getArguments())) {
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
        auto ret = dyn_cast<SPNReturn>(task.getBody().front().getTerminator());
        assert(ret);
        if (ret.getReturnValues().size() != task.getResults().size()) {
          return task.emitOpError() << "Task does not return the correct number of values";
        }
        for (auto retVal : llvm::zip(ret.getReturnValues(), task.getResults())) {
          if (std::get<0>(retVal).getType() != std::get<1>(retVal).getType()) {
            return task->emitOpError() << "Returned value type does not match Task result type";
          }
        }
        return mlir::success();
      }

      static mlir::LogicalResult verifyBody(SPNBody body) {
        // Check that the number and type of the entry block arguments match
        // the operands of the Body.
        if (body.getBody().front().getNumArguments() != body->getNumOperands()) {
          return body.emitOpError() << "Incorrect number of block arguments for entry block of Body";
        }
        for (auto argInput : llvm::zip(body.getBody().front().getArguments(), body.getInputs())) {
          auto argType = std::get<0>(argInput).getType();
          auto opType = std::get<1>(argInput).getType();
          // The low::LogType is only used inside SPNBody, so if the block argument is of LogType,
          // the operand should be of base-type.
          if (argType.isa<LogType>()) {
            argType = argType.cast<LogType>().getBaseType();
          }
          if (argType != opType) {
            return body.emitOpError() << "Body block argument type does not match Body operand type";
          }
        }
        // Check that the Body is terminated by a SPNYield with the correct number of return values and types.
        auto yield = dyn_cast<SPNYield>(body.getBody().front().getTerminator());
        assert(yield);
        if (yield.getResultValues().size() != body.getNumResults()) {
          return body.emitOpError() << "Body does not return the correct number of values";
        }
        for (auto retVal : llvm::zip(yield.getResultValues(), body->getResults())) {
          auto yieldResult = std::get<0>(retVal).getType();
          auto bodyResult = std::get<1>(retVal).getType();
          if (auto logType = yieldResult.dyn_cast<low::LogType>()) {
            // If the body internally computes in log-space, the body itself
            // will return a result corresponding to the base-type of the log-type,
            // as the log-type is only used internally to flag log-space computation.
            if (logType.getBaseType() != bodyResult) {
              return body.emitOpError() << "Log-type base type does not match Body result type";
            }
          } else {
            if (yieldResult != bodyResult) {
              return body.emitOpError() << "Returned value type does not match Body result type";
            }
          }
        }
        return mlir::success();
      }

      static mlir::LogicalResult verifyBatchExtract(SPNBatchExtract extract) {
        
        auto tensor = extract.getInput().getType().dyn_cast<TensorType>();
        assert(tensor);
        if (!tensor.hasRank()) {
          return extract.emitOpError() << "Input tensor should be ranked";
        }
        if (tensor.getRank() != 2) {
          return extract->emitOpError() << "Input tensor should be ranked with two dimensions";
        }
        unsigned staticDim = (extract.getTransposed().hasValue() && extract.getTransposed().getValue()) ? 0 : 1;
        if (tensor.isDynamicDim(staticDim)) {
          return extract->emitOpError() << "Dimension " << staticDim << " of input tensor should be static";
        }
        if (extract.getStaticIndex() >= tensor.getDimSize(staticDim)) {
          return extract.emitOpError() << "Static index out-of-bounds for input tensor";
        }
        if (tensor.getElementType() != extract.getResult().getType()) {
          return extract.emitOpError() << "Input tensor element type does not match output type";
        }
        return mlir::success();
      }

      static mlir::LogicalResult verifyBatchRead(SPNBatchRead read) {
        auto memref = read.getBatchMem().getType().dyn_cast<MemRefType>();
        assert(memref);
        if (!memref.hasRank()) {
          return read.emitOpError() << "Input memref should be ranked";
        }
        if (memref.getRank() != 2) {
          return read->emitOpError() << "Input memref should be ranked with two dimensions";
        }
        unsigned staticDim = (read.getTransposed().hasValue() && read.getTransposed().getValue()) ? 0 : 1;
        if (memref.isDynamicDim(staticDim)) {
          return read->emitOpError() << "Dimension " << staticDim << " of input memref should be static";
        }
        if (read.getStaticIndex() >= memref.getDimSize(staticDim)) {
          return read.emitOpError() << "Sample index out-of-bounds for input memref";
        }
        if (memref.getElementType() != read.getResult().getType()) {
          return read.emitOpError() << "Input memref element type does not match output type";
        }
        return mlir::success();
      }

      static mlir::LogicalResult verifyBatchCollect(SPNBatchCollect collect) {
        auto tensorTy = collect.getTensor().getType().dyn_cast<TensorType>();
        assert(tensorTy);

        if (!tensorTy.hasRank() || tensorTy.getRank() != 2) {
          return collect.emitOpError() << "Result tensor must be ranked with two dimensions";
        }
        unsigned staticDim = (collect.getTransposed().hasValue() && collect.getTransposed().getValue()) ? 0 : 1;
        if (tensorTy.isDynamicDim(staticDim) ||
            static_cast<long>(collect.getResultValues().size()) != tensorTy.getDimSize(staticDim)) {
          return collect.emitOpError() << "Result tensor's dimension "
                                       << staticDim << " must be static and match number of results";
        }
        auto elemTy = tensorTy.getElementType();
        auto mismatch = llvm::any_of(collect.getResultValues(), [elemTy](auto op) {
          return op.getType() != elemTy;
        });
        if (mismatch) {
          return collect.emitOpError() << "Scalar type and element type of tensor must match";
        }
        return mlir::success();
      }

      static mlir::LogicalResult verifyBatchWrite(SPNBatchWrite write) {
        auto memRefTy = write.getBatchMem().getType().dyn_cast<MemRefType>();
        assert(memRefTy);

        if (!memRefTy.hasRank() || memRefTy.getRank() != 2) {
          return write.emitOpError() << "Result memref must be ranked with two dimensions";
        }

        unsigned staticDim = (write.getTransposed().hasValue() && write.getTransposed().getValue()) ? 0 : 1;
        if (memRefTy.isDynamicDim(staticDim) ||
            static_cast<long>(write.getResultValues().size()) != memRefTy.getDimSize(staticDim)) {
          return write.emitOpError() << "Result memrefs's dimension "
                                     << staticDim << " must be static and match number of results";
        }

        auto elemTy = memRefTy.getElementType();
        auto mismatch = llvm::any_of(write.getResultValues(), [elemTy](auto op) {
          return op.getType() != elemTy;
        });
        if (mismatch) {
          return write.emitOpError() << "Scalar type and element type of memref must match";
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
  assert(getBody().empty() && "Task already has a block");
  auto* entry = new Block();
  getBody().push_back(entry);
  entry->addArgument(IndexType::get(this->getContext()));
  entry->addArguments(this->getInputs().getType());
  return entry;
}

mlir::Value mlir::spn::low::SPNTask::getBatchIndex() {
  assert(!getBody().empty() && "Task has no block");
  assert(getBody().front().getNumArguments() >= 1 && "Task block has no argument");
  return getBody().front().getArgument(0);
}

//===----------------------------------------------------------------------===//
// SPNBody
//===----------------------------------------------------------------------===//

mlir::Block* mlir::spn::low::SPNBody::addEntryBlock() {
  assert(getBody().empty() && "Body already has a block");
  auto* entry = new Block();
  getBody().push_back(entry);
  entry->addArguments(this->getInputs().getType());
  return entry;
}

//===----------------------------------------------------------------------===//
// SPNBatchRead
//===----------------------------------------------------------------------===//

void mlir::spn::low::SPNBatchRead::build(::mlir::OpBuilder& odsBuilder,
                                         ::mlir::OperationState& odsState,
                                         Value batchMem,
                                         Value dynamicIndex,
                                         unsigned int staticIndex,
                                         llvm::Optional<bool> transposed) {
  auto memrefTy = batchMem.getType().dyn_cast<MemRefType>();
  assert(memrefTy);
  auto resultTy = memrefTy.getElementType();
  auto staticIndexAttr = odsBuilder.getUI32IntegerAttr(staticIndex);
  bool transpose = transposed.hasValue() && transposed.getValue();
  build(odsBuilder, odsState, resultTy, batchMem, dynamicIndex, staticIndexAttr,
        odsBuilder.getBoolAttr(transpose));
}

//===----------------------------------------------------------------------===//
// SPNBatchCollect
//===----------------------------------------------------------------------===//

void mlir::spn::low::SPNBatchCollect::build(::mlir::OpBuilder& odsBuilder,
                                            ::mlir::OperationState& odsState,
                                            ValueRange resultValues,
                                            Value batchIndex,
                                            bool transposed) {
  auto numResults = static_cast<long>(resultValues.size());
  auto tensorTy = RankedTensorType::get({numResults, -1L}, resultValues.front().getType());
  build(odsBuilder, odsState, tensorTy, batchIndex, resultValues, odsBuilder.getBoolAttr(transposed));
}

//===----------------------------------------------------------------------===//
// SPNConstant
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult mlir::spn::low::SPNConstant::fold(::llvm::ArrayRef<::mlir::Attribute> operands) {
  assert(operands.empty() && "lo_spn.constant has no operands");
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// SPNLog
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult mlir::spn::low::SPNLog::fold(::llvm::ArrayRef<::mlir::Attribute> operands) {
  assert(operands.size() == 1 && "lospn.fold takes exactly one operand");
  if (operands[0]) {
    if (auto constOp = operands[0].dyn_cast<FloatAttr>()) {
      // If the input to the log is constant, we can replace the log with the
      // constant logarithm of the constant input.
      return FloatAttr::get(FloatType::getF64(this->getContext()),
                            std::log(constOp.getValueAsDouble()));
    }
  }
  // The single operand is not constant, return nullptr to signal that the operation
  // has not been touched.
  return nullptr;
}

//===----------------------------------------------------------------------===//
// SPNStripLog
//===----------------------------------------------------------------------===//

void mlir::spn::low::SPNStripLog::build(::mlir::OpBuilder& odsBuilder,
                                        ::mlir::OperationState& odsState,
                                        Value input,
                                        Type targetType) {
  build(odsBuilder, odsState, targetType, input, TypeAttr::get(targetType));
}

//===----------------------------------------------------------------------===//
// SPNConvertLog
//===----------------------------------------------------------------------===//

void mlir::spn::low::SPNConvertLog::build(::mlir::OpBuilder& odsBuilder,
                                          ::mlir::OperationState& odsState,
                                          Value input) {
  auto logType = mlir::spn::low::LogType::get(input.getType());
  build(odsBuilder, odsState, logType, input);
}

//===----------------------------------------------------------------------===//
// SPNMul
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult mlir::spn::low::SPNMul::fold(::llvm::ArrayRef<::mlir::Attribute> operands) {
  assert(operands.size() == 2 && "lospn.mul takes exactly two operands");
  if (operands[0] && operands[1]) {
    // Both operands are constant.
    auto lhs = operands[0].dyn_cast<FloatAttr>();
    auto rhs = operands[1].dyn_cast<FloatAttr>();
    // TODO If we want to support integer/fixed-point arithmetic, we also need to handle IntegerAttr.
    assert(lhs && rhs);
    return FloatAttr::get(lhs.getType(), lhs.getValueAsDouble() * rhs.getValueAsDouble());
  }
  // Constant operands of commutative operations are always moved to the right side of
  // the operation (see https://mlir.llvm.org/docs/Canonicalization/#globally-applied-rules),
  // so a check for a constant value on the right-hand side is sufficient.
  if (operands[1]) {
    auto rhs = operands[1].dyn_cast<FloatAttr>();
    // TODO If we want to support integer/fixed-point arithmetic, we also need to handle IntegerAttr.
    assert(rhs);
    // x * 0 == 0
    if (rhs.getValueAsDouble() == 0.0) {
      return FloatAttr::get(rhs.getType(), 0.0);
    }
    // x * 1 == x
    if (rhs.getValueAsDouble() == 1.0) {
      return this->getLeft();
    }
  }
  // None of the operands was constant, return nullptr to signal that the operations has not been touched.
  return nullptr;
}

//===----------------------------------------------------------------------===//
// SPNAdd
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult mlir::spn::low::SPNAdd::fold(::llvm::ArrayRef<::mlir::Attribute> operands) {
  assert(operands.size() == 2 && "lospn.add takes exactly two operands");
  if (operands[0] && operands[1]) {
    // Both operands are constant.
    auto lhs = operands[0].dyn_cast<FloatAttr>();
    auto rhs = operands[1].dyn_cast<FloatAttr>();
    // TODO If we want to support integer/fixed-point arithmetic, we also need to handle IntegerAttr.
    assert(lhs && rhs);
    return FloatAttr::get(lhs.getType(), lhs.getValueAsDouble() + rhs.getValueAsDouble());
  }
  // Constant operands of commutative operations are always moved to the right side of
  // the operation (see https://mlir.llvm.org/docs/Canonicalization/#globally-applied-rules),
  // so a check for a constant value on the right-hand side is sufficient.
  if (operands[1]) {
    auto rhs = operands[1].dyn_cast<FloatAttr>();
    // TODO If we want to support integer/fixed-point arithmetic, we also need to handle IntegerAttr.
    assert(rhs);
    // x + 0 == x
    if (rhs.getValueAsDouble() == 0.0) {
      return this->getLeft();
    }
  }
  // None of the operands was constant, return nullptr to signal that the operations has not been touched.
  return nullptr;
}

#define GET_OP_CLASSES
#include "LoSPN/LoSPNOps.cpp.inc"