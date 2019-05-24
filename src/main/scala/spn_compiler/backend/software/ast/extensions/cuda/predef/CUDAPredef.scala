package spn_compiler.backend.software.ast.extensions.cuda.predef

import spn_compiler.backend.software.ast.nodes.function.ASTExternalFunction
import spn_compiler.backend.software.ast.nodes.types.{ArrayType, IntegerType, StructType, VoidType}
import spn_compiler.backend.software.ast.nodes.value.constant.ASTConstant
import spn_compiler.backend.software.ast.nodes.variable.ASTVariable

case object CUDADim3Type extends StructType("dim3", List(("x", IntegerType), ("y", IntegerType), ("z", IntegerType)))

case object CUDAGridDim extends ASTVariable(CUDADim3Type, "gridDim")

case object CUDAGridID extends ASTVariable(CUDADim3Type, "gridId")

case object CUDABlockDim extends ASTVariable(CUDADim3Type, "blockDim")

case object CUDABlockID extends ASTVariable(CUDADim3Type, "blockId")

case object CUDAThreadID extends ASTVariable(CUDADim3Type, "threadId")

sealed abstract class CUDACopyDirection(val num : Int) extends ASTConstant(IntegerType, num)
case object CUDAMemCpyHostToHost extends CUDACopyDirection(0)
case object CUDAMemCpyHostToDevice extends CUDACopyDirection(1)
case object CUDAMemCpyDeviceToHost extends CUDACopyDirection(2)
case object CUDAMemCpyDeviceToDevice extends CUDACopyDirection(3)
case object CUDAMemCpyDefault extends CUDACopyDirection(4)


case object CUDAMemCpy extends ASTExternalFunction("cuda.h", "cudaMemcpy",
  IntegerType, ArrayType(VoidType),  ArrayType(VoidType), IntegerType, IntegerType)

case object CUDAMalloc extends ASTExternalFunction("cuda.h", "cudaMalloc", IntegerType,
  ArrayType(ArrayType(VoidType)), IntegerType)


