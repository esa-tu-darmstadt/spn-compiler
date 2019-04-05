package spn_compiler.backend.software.ast.nodes.value.expression

import spn_compiler.backend.software.ast.nodes.types.{ASTType, BooleanType}
import spn_compiler.backend.software.ast.nodes.value.ASTValue
import spn_compiler.backend.software.ast.nodes.value.expression.ASTBinaryExpression.CmpOpCode
import spn_compiler.backend.software.ast.nodes.value.expression.ASTCompareExpression._

abstract class ASTCompareExpression protected (left : ASTValue, right : ASTValue)
  extends ASTBinaryExpression(left, right) {

  val cmpCode : CmpCode

  override val opcode: ASTBinaryExpression.BinOpCode = CmpOpCode

  override def getType: ASTType = BooleanType
}

object ASTCompareExpression extends ASTBinaryExpressionExtractor(CmpOpCode) {
  sealed trait CmpCode
  case object EQCode extends CmpCode
  case object NECode extends CmpCode
  case object LTCode extends CmpCode
  case object LECode extends CmpCode
  case object GTCode extends CmpCode
  case object GECode extends CmpCode
}

abstract class ASTCompareExpressionExtractor(val cmpCode : CmpCode) {
  def unapply(arg: ASTCompareExpression): Option[(ASTValue, ASTValue)] = arg.cmpCode match {
    case `cmpCode` => Some(arg.leftOp, arg.rightOp)
    case _ => None
  }
}

class ASTCmpEQ private[ast](_leftOp : ASTValue, _rightOp : ASTValue) extends ASTCompareExpression(_leftOp, _rightOp){
  override val cmpCode: CmpCode = EQCode
}

object ASTCmpEQ extends ASTCompareExpressionExtractor(EQCode)

class ASTCmpNE private[ast](_leftOp : ASTValue, _rightOp : ASTValue) extends ASTCompareExpression(_leftOp, _rightOp){
  override val cmpCode: CmpCode = NECode
}

object ASTCmpNE extends ASTCompareExpressionExtractor(NECode)

class ASTCmpLT private[ast](_leftOp : ASTValue, _rightOp : ASTValue) extends ASTCompareExpression(_leftOp, _rightOp){
  override val cmpCode: CmpCode = LTCode
}

object ASTCmpLT extends ASTCompareExpressionExtractor(LTCode)

class ASTCmpLE private[ast](_leftOp : ASTValue, _rightOp : ASTValue) extends ASTCompareExpression(_leftOp, _rightOp){
  override val cmpCode: CmpCode = LECode
}

object ASTCmpLE extends ASTCompareExpressionExtractor(LECode)

class ASTCmpGT private[ast](_leftOp : ASTValue, _rightOp : ASTValue) extends ASTCompareExpression(_leftOp, _rightOp){
  override val cmpCode: CmpCode = GTCode
}

object ASTCmpGT extends ASTCompareExpressionExtractor(GTCode)

class ASTCmpGE private[ast](_leftOp : ASTValue, _rightOp : ASTValue) extends ASTCompareExpression(_leftOp, _rightOp){
  override val cmpCode: CmpCode = GECode
}

object ASTCmpGE extends ASTCompareExpressionExtractor(GECode)