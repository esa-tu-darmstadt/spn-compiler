package spn_compiler.backend.software.ast.nodes.value.type_conversion

import spn_compiler.backend.software.ast.nodes.types.ASTType
import spn_compiler.backend.software.ast.nodes.value.ASTValue

class ASTTypeConversion private[ast](val op : ASTValue, val targetType : ASTType) extends ASTValue {
  require(op.getType.convertible(targetType))

  override def getType: ASTType = targetType
}

object ASTTypeConversion {
  def unapply(arg: ASTTypeConversion): Option[(ASTValue, ASTType)] = Some(arg.op, arg.targetType)
}
