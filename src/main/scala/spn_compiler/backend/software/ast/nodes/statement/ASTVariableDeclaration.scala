package spn_compiler.backend.software.ast.nodes.statement

import spn_compiler.backend.software.ast.nodes.types.ASTType
import spn_compiler.backend.software.ast.nodes.variable.ASTVariable

class ASTVariableDeclaration[BaseType, VarType <: ASTType[BaseType]] private[ast](val variable : ASTVariable[BaseType, VarType])
  extends ASTStatement {

  variable.declaration = this

}

object ASTVariableDeclaration {

  def unappyl[BaseType, VarType <: ASTType[BaseType]](arg : ASTVariableDeclaration[BaseType, VarType])
    : Option[ASTVariable[BaseType, VarType]] = Some(arg.variable)

}