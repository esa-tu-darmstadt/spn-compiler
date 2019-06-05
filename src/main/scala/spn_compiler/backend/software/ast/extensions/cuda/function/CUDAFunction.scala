package spn_compiler.backend.software.ast.extensions.cuda.function

import spn_compiler.backend.software.ast.extensions.cuda.function.CUDAFunction.CUDAFunctionScope
import spn_compiler.backend.software.ast.nodes.function.{ASTFunction, ASTFunctionParameter}
import spn_compiler.backend.software.ast.nodes.types.ASTType

class CUDAFunction private[ast](val scope : CUDAFunctionScope, name : String, returnType : ASTType,
                                 params : ASTFunctionParameter*) extends ASTFunction(name, returnType, params:_*)

object CUDAFunction {

  sealed abstract class CUDAFunctionScope(val prefix : String)
  case object Host extends CUDAFunctionScope("__host__")
  case object Global extends CUDAFunctionScope("__global__")
  case object Device extends CUDAFunctionScope("__device__")

  def unapplySeq(func : CUDAFunction) : Option[(CUDAFunctionScope, String, ASTType, Seq[ASTFunctionParameter])] =
    Some(func.scope, func.name, func.returnType, func.getParameters.seq)

}