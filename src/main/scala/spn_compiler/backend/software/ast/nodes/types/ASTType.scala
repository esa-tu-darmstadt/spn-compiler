package spn_compiler.backend.software.ast.nodes.types

import scala.collection.immutable.ListMap

sealed abstract class ASTType{

  type BaseType

  def compatible(ty : ASTType) : Boolean = this.equals(ty)

  def isArrayType : Boolean = false

  def isStructType : Boolean = false

}

trait ScalarType extends ASTType

trait NumericType extends ScalarType

case object IntegerType extends NumericType {
  override type BaseType = Int
}

case object RealType extends NumericType {
  override type BaseType = Double
}

case object BooleanType extends ScalarType {
  override type BaseType = Boolean
}

case class ArrayType private[ast](elemType : ASTType) extends ASTType {
  override def isArrayType: Boolean = true
}

case class StructType private[ast](name : String, elements : List[(String, ASTType)]) extends ASTType {

  private val _elements : ListMap[String,  ASTType] = ListMap(elements :_*)

  def getElementType(id : String) : ASTType =
    _elements.getOrElse(id, throw new RuntimeException("No element with id %s found!".format(id)))

  override def isStructType: Boolean = true
}
