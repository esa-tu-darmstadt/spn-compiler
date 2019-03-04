package spn_compiler.backend.software.ast.nodes.statement.variable

import spn_compiler.backend.software.ast.nodes.reference.ASTReference
import spn_compiler.backend.software.ast.nodes.value.ASTValue

class ASTVariableAssignment private[ast](val lhs : ASTReference, val value : ASTValue) {

  require(lhs.getType==value.getType, "Type of the assigned value must match type of the variable!")

}

object ASTVariableAssignment {
  def unapply(arg: ASTVariableAssignment): Option[(ASTReference, ASTValue)] = Some(arg.lhs, arg.value)
}