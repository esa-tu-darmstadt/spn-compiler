package spn_compiler.backend.software.ast.nodes.value.constant

import spn_compiler.backend.software.ast.nodes.types.{ASTType, ArrayType, StructType}
import spn_compiler.backend.software.ast.nodes.value.ASTValue

class ASTArrayInit private[ast](val values : ASTValue*) extends ASTValue {
  private val arrayType = ArrayType(values.head.getType)

  require(values.forall(_.getType.compatible(arrayType.elemType)), "Element types must match array type!")

  override def getType: ASTType = arrayType
}

object ASTArrayInit {

  def unapplySeq(arg : ASTArrayInit) : Option[Seq[ASTValue]] = Some(arg.values)
}

class ASTStructInit private[ast](val structType: StructType, val values : ASTValue*) extends ASTValue {
  require((structType.elements.map(_._2) zip values.map(_.getType)).forall(t => t._1.compatible(t._2)),
  "Element types must match struct element types!")

  override def getType: ASTType = structType
}

object ASTStructInit {

  def unapplySeq(arg : ASTStructInit) : Option[(StructType, Seq[ASTValue])] = Some(arg.structType, arg.values)

}


