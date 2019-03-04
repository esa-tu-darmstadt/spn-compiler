package spn_compiler.backend.software.ast.nodes.value.expression

import spn_compiler.backend.software.ast.nodes.types.ASTType
import spn_compiler.backend.software.ast.nodes.value.ASTValue
import spn_compiler.backend.software.ast.nodes.value.expression.ASTUnaryExpression.{NegOpCode, NotOpCode, UnOpCode}

abstract class ASTUnaryExpression protected (val op : ASTValue) extends ASTValue {

  checkOperand()

  override def getType: ASTType = op.getType

  val opCode : ASTUnaryExpression.UnOpCode

  def checkOperand() : Unit = {
    require(op.getType.isScalarType)
  }
}

object ASTUnaryExpression {

  sealed trait UnOpCode
  case object NegOpCode extends UnOpCode
  case object NotOpCode extends UnOpCode

  def unapply(arg: ASTUnaryExpression): Option[ASTValue] = Some(arg.op)

}

abstract class ASTUnaryExpressionExtractor(val opCode : UnOpCode) {
  def unapply(arg: ASTUnaryExpression): Option[ASTValue] = arg.opCode match {
    case `opCode` => Some(arg.op)
    case _ => None
  }
}

class ASTNeg private[ast](_op : ASTValue) extends ASTUnaryExpression(_op){
  override val opCode: UnOpCode = NegOpCode
}

object ASTNeg extends ASTUnaryExpressionExtractor(NegOpCode)

trait ASTLogicUnaryExpression extends ASTUnaryExpression {
  override def checkOperand(): Unit = {
    super.checkOperand()
    op.getType.isLogicType
  }
}

class ASTNot private[ast](_op : ASTValue) extends ASTUnaryExpression(_op) with ASTLogicUnaryExpression {
  override val opCode: UnOpCode = NotOpCode
}

object ASTNot extends ASTUnaryExpressionExtractor(NotOpCode)