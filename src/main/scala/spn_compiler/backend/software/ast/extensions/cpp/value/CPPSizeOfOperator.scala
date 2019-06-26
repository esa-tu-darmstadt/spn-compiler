package spn_compiler.backend.software.ast.extensions.cpp.value

import spn_compiler.backend.software.ast.nodes.reference.ASTReference
import spn_compiler.backend.software.ast.nodes.types.{ASTType, IntegerType}
import spn_compiler.backend.software.ast.nodes.value.ASTValue

class CPPSizeOfOperator private[ast](val ref : ASTReference) extends ASTValue {
  override def getType: ASTType = IntegerType
}

object CPPSizeOfOperator {
  def unapply(arg: CPPSizeOfOperator): Option[ASTReference] = Some(arg.ref)
}
