package spn_compiler.backend.software.ast.nodes.value.expression

import spn_compiler.backend.software.ast.nodes.types.ASTType
import spn_compiler.backend.software.ast.nodes.value.ASTValue
import spn_compiler.backend.software.ast.nodes.value.expression.ASTBinaryExpression.AddOpCode

abstract class ASTBinaryExpression protected (val leftOp : ASTValue, val rightOp : ASTValue) extends ASTValue{

  checkOperands()

  override def getType: ASTType = leftOp.getType

  val opcode : ASTBinaryExpression.OpCode

  def checkOperands() : Unit = {
    require(leftOp.getType.compatible(rightOp.getType))
  }

}

object ASTBinaryExpression {
  sealed trait OpCode
  case object AddOpCode extends OpCode
  case object SubOpCode extends OpCode
  case object MulOpCode extends OpCode
  case object DivOpCode extends OpCode

  def unapply(arg: ASTBinaryExpression) : Option[(ASTValue, ASTValue)] = Some((arg.leftOp, arg.rightOp))

}


class ASTAddition private[ast] (_leftOp : ASTValue, _rightOp : ASTValue) extends ASTBinaryExpression(_leftOp, _rightOp){

  override val opcode: ASTBinaryExpression.OpCode = ASTBinaryExpression.AddOpCode

}

object ASTAddition {
  def unapply(arg : ASTBinaryExpression) : Option[(ASTValue, ASTValue)] = arg.opcode match{
    case AddOpCode => Some((arg.leftOp, arg.rightOp))
    case _ => None
  }
}
