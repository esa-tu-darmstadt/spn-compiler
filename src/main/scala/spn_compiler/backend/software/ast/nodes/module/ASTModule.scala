package spn_compiler.backend.software.ast.nodes.module

import spn_compiler.backend.software.ast.construct.{ASTBuilder, ASTTypeContext}
import spn_compiler.backend.software.ast.nodes.ASTNode
import spn_compiler.backend.software.ast.nodes.function.ASTFunction
import spn_compiler.backend.software.ast.nodes.types.StructType

class ASTModule(val name : String) extends ASTNode with ASTBuilder with ASTTypeContext

object ASTModule {

  def unapply(arg: ASTModule): Option[(String, List[StructType], List[ASTFunction])] =
    Some(arg.name, arg.structTypes.toList, arg.localFunctions.toList)

}
