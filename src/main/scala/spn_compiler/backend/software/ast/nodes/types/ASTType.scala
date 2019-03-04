package spn_compiler.backend.software.ast.nodes.types

import scala.collection.immutable.ListMap

sealed abstract class ASTType{

  type BaseType

  def compatible(ty : ASTType) : Boolean = this.equals(ty)

  // TODO: Currently all scalar types are convertible among each other. Issue warning for precision loss.
  def convertible(ty : ASTType) : Boolean = this.isScalarType && ty.isScalarType

  def isScalarType : Boolean = false

  def isArrayType : Boolean = false

  def isStructType : Boolean = false

  def isLogicType : Boolean = false

}

trait ScalarType extends ASTType {
  override def isScalarType: Boolean = true
}

trait NumericType extends ScalarType

trait LogicType extends ScalarType {
  override def isLogicType: Boolean = true
}

case object IntegerType extends NumericType with LogicType {
  override type BaseType = Int
}

case object RealType extends NumericType {
  override type BaseType = Double
}

case object BooleanType extends ScalarType with LogicType {
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
