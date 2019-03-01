package spn_compiler.backend.software.ast.nodes.statement

import spn_compiler.backend.software.ast.nodes.variable.ASTVariable

class ASTVariableDeclaration private[ast](val variable : ASTVariable)
  extends ASTStatement {

  variable.declaration = this

}

object ASTVariableDeclaration {

  def unapply(arg : ASTVariableDeclaration)
    : Option[ASTVariable] = Some(arg.variable)

}