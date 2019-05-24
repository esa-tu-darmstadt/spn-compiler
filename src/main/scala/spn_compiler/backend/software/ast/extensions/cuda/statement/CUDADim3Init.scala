package spn_compiler.backend.software.ast.extensions.cuda.statement

import spn_compiler.backend.software.ast.extensions.cuda.predef.CUDADim3Type
import spn_compiler.backend.software.ast.nodes.statement.ASTStatement
import spn_compiler.backend.software.ast.nodes.types.IntegerType
import spn_compiler.backend.software.ast.nodes.value.ASTValue
import spn_compiler.backend.software.ast.nodes.value.constant.ASTConstant
import spn_compiler.backend.software.ast.nodes.variable.ASTVariable

class CUDADim3Init private[ast](val variable : ASTVariable,
                                val x : ASTValue = new ASTConstant(IntegerType, 1),
                                val y : ASTValue = new ASTConstant(IntegerType, 1),
                                val z : ASTValue = new ASTConstant(IntegerType, 1)) extends ASTStatement{
  require(variable.getType == CUDADim3Type)
  require(x.getType == IntegerType)
  require(y.getType == IntegerType)
  require(z.getType == IntegerType)

}

object CUDADim3Init {

  def unapply(arg: CUDADim3Init): Option[(ASTVariable, ASTValue, ASTValue, ASTValue)] =
    Some(arg.variable,arg.x, arg.y, arg.z)

}