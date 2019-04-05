package spn_compiler.backend.software.ast.nodes.value.constant

import spn_compiler.backend.software.ast.nodes.types.ScalarType
import spn_compiler.backend.software.ast.nodes.value.ASTValue

class ASTConstant[ConstantType <: ScalarType, BaseType <: ConstantType#BaseType] private[ast]
  (private val ty : ConstantType, private val value : BaseType) extends ASTValue {

  def getConstantValue : BaseType = value

  override def getType: ConstantType = ty

}

object ASTConstant {

  def unapply[ConstantType <: ScalarType, BaseType <: ConstantType#BaseType](arg: ASTConstant[ConstantType, BaseType]) :
    Option[(ConstantType, BaseType)] = Some(arg.ty, arg.value)

}
