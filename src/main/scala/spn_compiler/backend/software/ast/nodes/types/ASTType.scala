package spn_compiler.backend.software.ast.nodes.types

import scala.collection.immutable.ListMap

sealed abstract class ASTType{

  type BaseType

  def compatible(ty : ASTType) : Boolean = this.equals(ty)

  // TODO: Currently all scalar types are convertible among each other. Issue warning for precision loss.
  def convertible(ty : ASTType) : Boolean =
    (this.isScalarType && ty.isScalarType) || (this.isArrayType && ty.isArrayType)

  def isScalarType : Boolean = false

  def isArrayType : Boolean = false

  def isStructType : Boolean = false

  def isLogicType : Boolean = false

}

case object VoidType extends ASTType

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

case object PreciseIntegerType extends NumericType with LogicType {
  override type BaseType = String
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

class StructType private[ast](val name : String, val elements : List[(String, ASTType)]) extends ASTType {

  private val _elements : ListMap[String,  ASTType] = ListMap(elements :_*)

  def getElementType(id : String) : ASTType =
    _elements.getOrElse(id, throw new RuntimeException("No element with id %s found!".format(id)))

  override def isStructType: Boolean = true
}

object StructType {

  def unapplySeq(arg: StructType): Option[(String, Seq[(String, ASTType)])] = Some(arg.name, arg.elements)

}

abstract class EnumBaseType {
  def toString : String
}

class EnumType [B <: EnumBaseType] private[ast](enumValues : B*) extends ScalarType {
  override type BaseType = B
}