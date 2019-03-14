package spn_compiler.backend.software.ast.nodes.reference

import spn_compiler.backend.software.ast.nodes.ASTNode
import spn_compiler.backend.software.ast.nodes.types.{ASTType, ArrayType, StructType}
import spn_compiler.backend.software.ast.nodes.value.ASTValue

trait ASTReferencable extends ASTNode {

  def getType : ASTType

}

trait ASTReference extends ASTNode {

  def getType : ASTType

}

class ASTVariableReference private[ast](val variable : ASTReferencable) extends ASTReference {

  def getType : ASTType = variable.getType

}

object ASTVariableReference {

  def unapply(arg : ASTVariableReference) : Option[(ASTType, ASTReferencable)] = Some(arg.getType, arg.variable)

}

class ASTIndexReference private[ast](val reference : ASTReference, val index : ASTValue) extends ASTReference {
  require(reference.getType.isArrayType, "Can only reference an array by index!")

  override def getType: ASTType = reference.getType.asInstanceOf[ArrayType].elemType
}

object ASTIndexReference {

  def unapply(arg : ASTIndexReference) : Option[(ASTType, ASTReference, ASTValue)] =
    Some(arg.getType, arg.reference, arg.index)

}

class ASTElementReference private[ast](val reference : ASTReference, val element : String) extends ASTReference {
  require(reference.getType.isStructType, "Can only reference elements of a struct by name!")

  override def getType: ASTType = reference.getType.asInstanceOf[StructType].getElementType(element)
}

object ASTElementReference {

  def unapply(arg: ASTElementReference): Option[(ASTType, ASTReference, String)] =
    Some(arg.getType, arg.reference, arg.element)

}
