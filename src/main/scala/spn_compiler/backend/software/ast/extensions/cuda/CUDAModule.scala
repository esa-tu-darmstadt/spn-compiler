package spn_compiler.backend.software.ast.extensions.cuda

import spn_compiler.backend.software.ast.nodes.function.ASTFunction
import spn_compiler.backend.software.ast.nodes.module.ASTModule
import spn_compiler.backend.software.ast.nodes.statement.variable.ASTVariableDeclaration
import spn_compiler.backend.software.ast.nodes.types.StructType

class CUDAModule(name : String) extends ASTModule(name) with CUDAASTBuilder

object CUDAModule {
  def unapply(arg: CUDAModule): Option[(String, List[StructType], List[ASTVariableDeclaration], List[ASTFunction])] =
    Some(arg.name, arg.structTypes.toList, arg.globalVariables.toList, arg.localFunctions.toList)
}
