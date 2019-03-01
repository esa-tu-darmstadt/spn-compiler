package spn_compiler.backend.software.ast.nodes.value

import spn_compiler.backend.software.ast.nodes.types.ASTType

abstract class ASTValue {

  def getType : ASTType

}
