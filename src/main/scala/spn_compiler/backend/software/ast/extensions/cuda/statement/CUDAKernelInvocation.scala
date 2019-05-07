package spn_compiler.backend.software.ast.extensions.cuda.statement

import spn_compiler.backend.software.ast.extensions.cuda.function.CUDAFunction
import spn_compiler.backend.software.ast.extensions.cuda.predef.CUDADim3Type
import spn_compiler.backend.software.ast.nodes.statement.ASTStatement
import spn_compiler.backend.software.ast.nodes.value.ASTValue

class CUDAKernelInvocation private[ast](val gridLayout : ASTValue, val blockLayout : ASTValue, val kernel: CUDAFunction,
                           val params : ASTValue*) extends ASTStatement {
  require(gridLayout.getType == CUDADim3Type)
  require(blockLayout.getType == CUDADim3Type)
  require(kernel.scope == CUDAFunction.Global) // For a kernel to be called from the host, it must be in global scope.
  require((kernel.getParameterTypes zip params.map(_.getType)).forall{case (pt, at) => pt == at},
    "Actual parameter types must match formal parameter types!")
}

object CUDAKernelInvocation {

  def unapplySeq(invoke : CUDAKernelInvocation) : Option[(ASTValue, ASTValue, CUDAFunction, Seq[ASTValue])] =
    Some(invoke.gridLayout, invoke.blockLayout, invoke.kernel, invoke.params)

}
