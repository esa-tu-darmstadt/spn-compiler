package spn_compiler.backend.software.ast.nodes.variable

import spn_compiler.backend.software.ast.nodes.ASTNode
import spn_compiler.backend.software.ast.nodes.statement.ASTVariableDeclaration
import spn_compiler.backend.software.ast.nodes.types.ASTType

class ASTVariable[BaseType, VarType <: ASTType[BaseType]] private[ast]
  (val ty : VarType, val name : String) extends ASTNode {

  private var _declaration : Option[ASTVariableDeclaration[BaseType, VarType]] = None

  def isDeclared : Boolean = _declaration.isDefined

  def declaration : ASTVariableDeclaration[BaseType, VarType] = _declaration.orNull

  def declaration_=(declare : ASTVariableDeclaration[BaseType, VarType]) : Unit = _declaration = Some(declare)

}

object ASTVariable {

  def unapply[BaseType, VarType <: ASTType[BaseType]](arg : ASTVariable[BaseType, VarType])
    : Option[(VarType, String)] = Some((arg.ty, arg.name))

}