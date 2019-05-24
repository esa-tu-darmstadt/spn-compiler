package spn_compiler.backend.software.ast.extensions.cpp.value

import spn_compiler.backend.software.ast.nodes.reference.ASTReference
import spn_compiler.backend.software.ast.nodes.types.{ASTType, ArrayType}
import spn_compiler.backend.software.ast.nodes.value.ASTValue

class CPPAddressOfOperator private[ast](val ref : ASTReference) extends ASTValue {
  override def getType: ASTType = ArrayType(ref.getType)
}

object CPPAddressOfOperator {

  def unapply(arg: CPPAddressOfOperator): Option[ASTReference] = Some(arg.ref)

}
