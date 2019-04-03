package spn_compiler.backend.software.ast.nodes.statement.variable

import spn_compiler.backend.software.ast.nodes.statement.ASTStatement
import spn_compiler.backend.software.ast.nodes.value.ASTValue
import spn_compiler.backend.software.ast.nodes.variable.ASTVariable

class ASTVariableDeclaration private[ast](val variable : ASTVariable, val initValue : Option[ASTValue] = None)
  extends ASTStatement {

  variable.declaration = this

}

object ASTVariableDeclaration {

  def unapply(arg : ASTVariableDeclaration)
    : Option[(ASTVariable, Option[ASTValue])] = Some(arg.variable, arg.initValue)

}