package spn_compiler.backend.software.ast.nodes.value.constant

import spn_compiler.backend.software.ast.nodes.types.ScalarType
import spn_compiler.backend.software.ast.nodes.value.ASTValue

class ASTConstant[BaseType, ConstantType <: ScalarType[BaseType]] private[ast] (private val ty : ConstantType, private val value : BaseType)
  extends ASTValue[BaseType, ConstantType] {

  def getConstantValue : BaseType = value

  override def getType: ConstantType = ty

}

object ASTConstant {

  def unapply[BaseType, ConstantType <: ScalarType[BaseType]](arg: ASTConstant[BaseType, ConstantType]) :
    Option[(ScalarType[BaseType], BaseType)] = Some(arg.ty, arg.value)

}
