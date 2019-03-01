package spn_compiler.backend.software.ast.nodes.types

sealed abstract class ASTType{

  type BaseType

  def compatible(ty : ASTType) : Boolean = this.equals(ty)

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

case class ArrayType(elemType : ASTType) extends ASTType

