package spn_compiler.backend.software.ast.extensions.cuda.predef

import spn_compiler.backend.software.ast.extensions.cuda.predef.CUDAMemCpyKind.{Device2Device, Device2Host, Host2Device, Host2Host}
import spn_compiler.backend.software.ast.nodes.function.ASTExternalFunction
import spn_compiler.backend.software.ast.nodes.types._
import spn_compiler.backend.software.ast.nodes.value.constant.ASTConstant
import spn_compiler.backend.software.ast.nodes.variable.ASTVariable

case object CUDADim3Type extends StructType("dim3", List(("x", IntegerType), ("y", IntegerType), ("z", IntegerType)))

case object CUDAGridDim extends ASTVariable(CUDADim3Type, "gridDim")

case object CUDAGridID extends ASTVariable(CUDADim3Type, "gridIdx")

case object CUDABlockDim extends ASTVariable(CUDADim3Type, "blockDim")

case object CUDABlockID extends ASTVariable(CUDADim3Type, "blockIdx")

case object CUDAThreadID extends ASTVariable(CUDADim3Type, "threadIdx")



object CUDAMemCpyKind {
  sealed trait CUDACopyDirection extends EnumBaseType
  case object Host2Host extends CUDACopyDirection {
    override def toString: String = "cudaMemcpyHostToHost"
  }
  case object Host2Device extends CUDACopyDirection {
    override def toString: String = "cudaMemcpyHostToDevice"
  }

  case object Device2Host extends CUDACopyDirection {
    override def toString: String = "cudaMemcpyDeviceToHost"
  }
  case object Device2Device extends CUDACopyDirection {
    override def toString: String = "cudaMemcpyDeviceToDevice"
  }
  val enumType = new EnumType[CUDACopyDirection](Host2Host, Host2Device, Device2Host, Device2Device)
}

sealed abstract class CUDAMemCpyKind(val kind : CUDAMemCpyKind.CUDACopyDirection)
  extends ASTConstant(CUDAMemCpyKind.enumType, kind)
case object CUDAMemCpyHostToHost extends CUDAMemCpyKind(Host2Host)
case object CUDAMemCpyHostToDevice extends CUDAMemCpyKind(Host2Device)
case object CUDAMemCpyDeviceToHost extends CUDAMemCpyKind(Device2Host)
case object CUDAMemCpyDeviceToDevice extends CUDAMemCpyKind(Device2Device)

case object CUDAMemCpy extends ASTExternalFunction("cuda.h", "cudaMemcpy",
  IntegerType, ArrayType(VoidType),  ArrayType(VoidType), IntegerType, CUDAMemCpyKind.enumType)

case object CUDAMalloc extends ASTExternalFunction("cuda.h", "cudaMalloc", IntegerType,
  ArrayType(ArrayType(VoidType)), IntegerType)


