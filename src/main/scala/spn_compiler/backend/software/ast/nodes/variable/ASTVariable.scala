package spn_compiler.backend.software.ast.nodes.variable

import spn_compiler.backend.software.ast.nodes.ASTNode
import spn_compiler.backend.software.ast.nodes.statement.ASTVariableDeclaration
import spn_compiler.backend.software.ast.nodes.types.ASTType

class ASTVariable private[ast]
  (val ty : ASTType, val name : String) extends ASTNode {

  private var _declaration : Option[ASTVariableDeclaration] = None

  def isDeclared : Boolean = _declaration.isDefined

  def declaration : ASTVariableDeclaration = _declaration.orNull

  def declaration_=(declare : ASTVariableDeclaration) : Unit = _declaration = Some(declare)

}

object ASTVariable {

  def unapply(arg : ASTVariable)
    : Option[(ASTType, String)] = Some((arg.ty, arg.name))

}