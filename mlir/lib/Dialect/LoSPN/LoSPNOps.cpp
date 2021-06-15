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

      static mlir::LogicalResult verifyBody(SPNBody body) {
        // Check that the number and type of the entry block arguments match
        // the operands of the Body.
        if (body.body().front().getNumArguments() != body->getNumOperands()) {
          return body.emitOpError() << "Incorrect number of block arguments for entry block of Body";
        }
        for (auto argInput : llvm::zip(body.body().front().getArguments(), body.inputs())) {
          if (std::get<0>(argInput).getType() != std::get<1>(argInput).getType()) {
            return body.emitOpError() << "Body block argument type does not match Body operand type";
          }
        }
        // Check that the Body is terminated by a SPNYield with the correct number of return values and types.
        auto yield = dyn_cast<SPNYield>(body.body().front().getTerminator());
        assert(yield);
        if (yield.resultValues().size() != body.getNumResults()) {
          return body.emitOpError() << "Body does not return the correct number of values";
        }
        for (auto retVal : llvm::zip(yield.resultValues(), body->getResults())) {
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
        auto tensor = extract.input().getType().dyn_cast<TensorType>();
        assert(tensor);
        if (!tensor.hasRank() || tensor.getRank() != 2) {
          return extract->emitOpError() << "Input tensor should be ranked with two dimensions";
        }
        if (tensor.isDynamicDim(1)) {
          return extract->emitOpError() << "Second dimension of input tensor should be static";
        }
        if (extract.sampleIndex() >= tensor.getDimSize(1)) {
          return extract.emitOpError() << "Sample index out-of-bounds for input tensor";
        }
        if (tensor.getElementType() != extract.result().getType()) {
          return extract.emitOpError() << "Input tensor element type does not match output type";
        }
        return mlir::success();
      }

      static mlir::LogicalResult verifyBatchRead(SPNBatchRead read) {
        auto memref = read.batchMem().getType().dyn_cast<MemRefType>();
        assert(memref);
        if (!memref.hasRank() || memref.getRank() != 2) {
          return read->emitOpError() << "Input memref should be ranked with two dimensions";
        }
        if (memref.isDynamicDim(1)) {
          return read->emitOpError() << "Second dimension of input memref should be static";
        }
        if (read.sampleIndex() >= memref.getDimSize(1)) {
          return read.emitOpError() << "Sample index out-of-bounds for input memref";
        }
        if (memref.getElementType() != read.result().getType()) {
          return read.emitOpError() << "Input memref element type does not match output type";
        }
        return mlir::success();
      }

      static mlir::LogicalResult verifyBatchCollect(SPNBatchCollect collect) {
        if (collect.resultValues().size() != collect->getNumResults()) {
          return collect.emitOpError() << "Number of scalar result values must match the number of returned tensors";
        }
        for (auto scalarAndTensor : llvm::zip(collect.resultValues(), collect.tensors())) {
          auto tensorTy = std::get<1>(scalarAndTensor).getType().dyn_cast<TensorType>();
          assert(tensorTy);
          if (std::get<0>(scalarAndTensor).getType() != tensorTy.getElementType()) {
            return collect.emitOpError() << "Scalar type and element type of tensor must match";
          }
        }
        return mlir::success();
      }

      static mlir::LogicalResult verifyBatchWrite(SPNBatchWrite write) {
        auto memRefTy = write.batchMem().getType().dyn_cast<MemRefType>();
        assert(memRefTy);
        if (write.resultValue().getType() != memRefTy.getElementType()) {
          return write.emitOpError() << "Scalar type and element type of result memory must match";
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
// SPNTask
//===----------------------------------------------------------------------===//

mlir::Block* mlir::spn::low::SPNBody::addEntryBlock() {
  assert(body().empty() && "Body already has a block");
  auto* entry = new Block();
  body().push_back(entry);
  entry->addArguments(this->inputs().getType());
  return entry;
}

//===----------------------------------------------------------------------===//
// SPNConstant
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult mlir::spn::low::SPNConstant::fold(::llvm::ArrayRef<::mlir::Attribute> operands) {
  assert(operands.empty() && "lo_spn.constant has no operands");
  return valueAttr();
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
      return this->left();
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
      return this->left();
    }
  }
  // None of the operands was constant, return nullptr to signal that the operations has not been touched.
  return nullptr;
}

//===----------------------------------------------------------------------===//
// SPNCategoricalLeaf
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::spn::low::SPNCategoricalLeaf::canonicalize(SPNCategoricalLeaf op,
                                                                       PatternRewriter& rewriter) {
  // Rewrite Categoricals which contain exactly two probabilities into a LoSPN Select.
  auto probabilities = op.probabilities().getValue();
  if (probabilities.size() == 2) {
    auto pTrue = probabilities[0].dyn_cast<FloatAttr>();
    auto pFalse = probabilities[1].dyn_cast<FloatAttr>();
    auto threshold_max_true = FloatAttr::get(op.index().getType(), 1.0);
    rewriter.replaceOpWithNewOp<SPNSelectLeaf>(op,
                                               pTrue.getType(),
                                               op.index(),
                                               threshold_max_true,
                                               pTrue,
                                               pFalse,
                                               op.supportMarginalAttr());
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// SPNHistogramLeaf
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::spn::low::SPNHistogramLeaf::canonicalize(SPNHistogramLeaf op, PatternRewriter& rewriter) {
  // Rewrite certain Histograms which contain exactly two buckets into a LoSPN Select.
  // Buckets' index range must be 1 and buckets have to be consecutive / contiguous.
  // i.e.: (UB_0-LB_0 == 1) && (UB_1-LB_1 == 1) && (UB_0 == LB_1)
  auto buckets = op.buckets();
  if (buckets.size() == 2) {
    auto b0 = buckets[0].cast<mlir::spn::low::Bucket>();
    auto b1 = buckets[1].cast<mlir::spn::low::Bucket>();

    bool isQualifiedIndexRange = ((b0.ub().getInt() - b0.lb().getInt()) == 1) &&
        ((b1.ub().getInt() - b1.lb().getInt()) == 1);
    bool isContiguous = (b0.ub().getInt() == b1.lb().getInt());

    if (isQualifiedIndexRange && isContiguous) {
      auto pTrue = b0.val();
      auto pFalse = b1.val();
      auto threshold_max_true = FloatAttr::get(Float64Type::get(op.getContext()), b0.ub().getInt());
      rewriter.replaceOpWithNewOp<SPNSelectLeaf>(op,
                                                 pTrue.getType(),
                                                 op.index(),
                                                 threshold_max_true,
                                                 pTrue,
                                                 pFalse,
                                                 op.supportMarginalAttr());
      return success();
    }
  }
  return failure();
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
    if (auto outputLogType = this->getResult().getType().dyn_cast<low::LogType>()) {
      if (auto logFloatType = outputLogType.getBaseType().dyn_cast<FloatType>()) {
        if (inputFloatType.getWidth() != logFloatType.getWidth()) {
          return false;
        }
      }
    }
  }
  return true;
}

#define GET_OP_CLASSES
#include "LoSPN/LoSPNOps.cpp.inc"