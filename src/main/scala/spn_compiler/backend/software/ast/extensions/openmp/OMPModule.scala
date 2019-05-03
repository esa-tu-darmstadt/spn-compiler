package spn_compiler.backend.software.ast.extensions.openmp

import spn_compiler.backend.software.ast.nodes.function.ASTFunction
import spn_compiler.backend.software.ast.nodes.module.ASTModule
import spn_compiler.backend.software.ast.nodes.statement.variable.ASTVariableDeclaration
import spn_compiler.backend.software.ast.nodes.types.StructType

class OMPModule(name : String) extends ASTModule(name) with OMPASTBuilder

object OMPModule {

  def unapply(arg: OMPModule): Option[(String, List[StructType], List[ASTVariableDeclaration], List[ASTFunction])] =
    Some(arg.name, arg.structTypes.toList, arg.globalVariables.toList, arg.localFunctions.toList)

}