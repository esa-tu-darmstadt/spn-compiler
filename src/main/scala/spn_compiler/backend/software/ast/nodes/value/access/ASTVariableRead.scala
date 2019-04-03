package spn_compiler.backend.software.ast.nodes.value.access

import spn_compiler.backend.software.ast.nodes.reference.ASTReference
import spn_compiler.backend.software.ast.nodes.types.ASTType
import spn_compiler.backend.software.ast.nodes.value.ASTValue

class ASTVariableRead private[ast](val reference : ASTReference) extends ASTValue {
  override def getType: ASTType = reference.getType
}

object ASTVariableRead {

  def unapply(arg: ASTVariableRead): Option[(ASTType, ASTReference)] = Some(arg.getType, arg.reference)

}
