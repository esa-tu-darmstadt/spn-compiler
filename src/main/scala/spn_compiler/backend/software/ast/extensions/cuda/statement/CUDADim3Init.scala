package spn_compiler.backend.software.ast.extensions.cuda.statement

import spn_compiler.backend.software.ast.nodes.statement.ASTStatement
import spn_compiler.backend.software.ast.nodes.types.IntegerType
import spn_compiler.backend.software.ast.nodes.value.ASTValue
import spn_compiler.backend.software.ast.nodes.value.constant.ASTConstant

class CUDADim3Init private[ast](val x : ASTValue = new ASTConstant(IntegerType, 1),
                                val y : ASTValue = new ASTConstant(IntegerType, 1),
                                val z : ASTValue = new ASTConstant(IntegerType, 1)) extends ASTStatement{

  require(x.getType == IntegerType)
  require(y.getType == IntegerType)
  require(z.getType == IntegerType)

}

object CUDADim3Init {

  def unapply(arg: CUDADim3Init): Option[(ASTValue, ASTValue, ASTValue)] = Some(arg.x, arg.y, arg.z)

}