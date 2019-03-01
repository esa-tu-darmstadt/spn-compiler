package spn_compiler.backend.software.ast.nodes.value.expression

import spn_compiler.backend.software.ast.nodes.types.{ASTType, NumericType}
import spn_compiler.backend.software.ast.nodes.value.ASTValue
import spn_compiler.backend.software.ast.nodes.value.expression.ASTBinaryExpression.AddOpCode

abstract class ASTBinaryExpression[BaseType, ValueType <: ASTType[BaseType]] protected
  (val leftOp : ASTValue[BaseType, ValueType], val rightOp : ASTValue[BaseType, ValueType])
  extends ASTValue[BaseType, ValueType]{

  val opcode : ASTBinaryExpression.OpCode

  override def getType: ASTType[BaseType] = leftOp.getType
}

object ASTBinaryExpression {
  sealed trait OpCode
  case object AddOpCode extends OpCode
  case object SubOpCode extends OpCode
  case object MulOpCode extends OpCode
  case object DivOpCode extends OpCode

  def unapply[BaseType, ValueType <: ASTType[BaseType]](arg: ASTBinaryExpression[BaseType, ValueType]) :
    Option[(ASTValue[BaseType, ValueType], ASTValue[BaseType, ValueType])] = Some((arg.leftOp, arg.rightOp))

}

class ASTAddition[BaseType, ValueType <: NumericType[BaseType]] private[ast]
  (_leftOp : ASTValue[BaseType, ValueType], _rightOp : ASTValue[BaseType, ValueType])
  extends ASTBinaryExpression[BaseType, ValueType](_leftOp, _rightOp){

  override val opcode: ASTBinaryExpression.OpCode = ASTBinaryExpression.AddOpCode

}

object ASTAddition {
  def unapply[BaseType, ValueType <: NumericType[BaseType]](arg : ASTBinaryExpression[BaseType, ValueType]) :
    Option[(ASTValue[BaseType, ValueType], ASTValue[BaseType, ValueType])] = arg.opcode match{
    case AddOpCode => Some((arg.leftOp, arg.rightOp))
    case _ => None
  }
}
