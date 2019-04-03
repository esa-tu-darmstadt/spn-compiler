package spn_compiler.backend.software.ast.nodes.value.expression

import spn_compiler.backend.software.ast.nodes.types.ASTType
import spn_compiler.backend.software.ast.nodes.value.ASTValue
import spn_compiler.backend.software.ast.nodes.value.expression.ASTBinaryExpression._

abstract class ASTBinaryExpression protected (val leftOp : ASTValue, val rightOp : ASTValue) extends ASTValue{

  checkOperands()

  override def getType: ASTType = leftOp.getType

  val opcode : ASTBinaryExpression.BinOpCode

  def checkOperands() : Unit = {
    require(leftOp.getType.compatible(rightOp.getType))
    require(leftOp.getType.isScalarType)
    require(rightOp.getType.isScalarType)
  }

}

object ASTBinaryExpression {
  sealed trait BinOpCode
  case object AddOpCode extends BinOpCode
  case object SubOpCode extends BinOpCode
  case object MulOpCode extends BinOpCode
  case object DivOpCode extends BinOpCode
  case object RemOpCode extends BinOpCode
  case object AndOpCode extends BinOpCode
  case object  OrOpCode extends BinOpCode
  case object XorOpCode extends BinOpCode
  case object CmpOpCode extends BinOpCode

  def unapply(arg: ASTBinaryExpression) : Option[(ASTValue, ASTValue)] = Some((arg.leftOp, arg.rightOp))

}

abstract class ASTBinaryExpressionExtractor(val opCode : BinOpCode) {

  def unapply(arg : ASTBinaryExpression) : Option[(ASTValue, ASTValue)] = arg.opcode match {
    case `opCode` => Some(arg.leftOp, arg.rightOp)
    case _ => None
  }
}

class ASTAddition private[ast] (_leftOp : ASTValue, _rightOp : ASTValue) extends ASTBinaryExpression(_leftOp, _rightOp){
  override val opcode: ASTBinaryExpression.BinOpCode = ASTBinaryExpression.AddOpCode
}

object ASTAddition extends ASTBinaryExpressionExtractor(AddOpCode)

class ASTSubtraction private[ast] (_leftOp : ASTValue, _rightOp : ASTValue) extends ASTBinaryExpression(_leftOp, _rightOp){
  override val opcode: ASTBinaryExpression.BinOpCode = ASTBinaryExpression.SubOpCode
}

object ASTSubtraction extends ASTBinaryExpressionExtractor(SubOpCode)

class ASTMultiplication private[ast] (_leftOp : ASTValue, _rightOp : ASTValue) extends ASTBinaryExpression(_leftOp, _rightOp){
  override val opcode: ASTBinaryExpression.BinOpCode = ASTBinaryExpression.MulOpCode
}

object ASTMultiplication extends ASTBinaryExpressionExtractor(MulOpCode)

class ASTDivision private[ast] (_leftOp : ASTValue, _rightOp : ASTValue) extends ASTBinaryExpression(_leftOp, _rightOp){
  override val opcode: ASTBinaryExpression.BinOpCode = ASTBinaryExpression.DivOpCode
}

object ASTDivision extends ASTBinaryExpressionExtractor(DivOpCode)

class ASTRemainder private[ast] (_leftOp : ASTValue, _rightOp : ASTValue) extends ASTBinaryExpression(_leftOp, _rightOp){
  override val opcode: ASTBinaryExpression.BinOpCode = ASTBinaryExpression.RemOpCode
}

object ASTRemainder extends ASTBinaryExpressionExtractor(RemOpCode)

trait ASTLogicBinaryExpression extends ASTBinaryExpression {
  override def checkOperands(): Unit = {
    super.checkOperands()
    require(leftOp.getType.isLogicType)
    require(rightOp.getType.isLogicType)
  }
}

class ASTAnd private[ast] (_leftOp : ASTValue, _rightOp : ASTValue) extends ASTBinaryExpression(_leftOp, _rightOp)
  with ASTLogicBinaryExpression {
  override val opcode: ASTBinaryExpression.BinOpCode = ASTBinaryExpression.AndOpCode
}

object ASTAnd extends ASTBinaryExpressionExtractor(AndOpCode)

class ASTOr private[ast] (_leftOp : ASTValue, _rightOp : ASTValue) extends ASTBinaryExpression(_leftOp, _rightOp)
  with ASTLogicBinaryExpression {
  override val opcode: ASTBinaryExpression.BinOpCode = ASTBinaryExpression.OrOpCode
}

object ASTOr extends ASTBinaryExpressionExtractor(OrOpCode)

class ASTXor private[ast] (_leftOp : ASTValue, _rightOp : ASTValue) extends ASTBinaryExpression(_leftOp, _rightOp)
  with ASTLogicBinaryExpression {
  override val opcode: ASTBinaryExpression.BinOpCode = ASTBinaryExpression.XorOpCode
}

object ASTXor extends ASTBinaryExpressionExtractor(XorOpCode)