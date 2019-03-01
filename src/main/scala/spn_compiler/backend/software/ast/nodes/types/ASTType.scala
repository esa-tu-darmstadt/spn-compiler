package spn_compiler.backend.software.ast.nodes.types

sealed abstract class ASTType[BaseType]

sealed abstract class ScalarType[BaseType] extends ASTType[BaseType]

sealed abstract class NumericType[BaseType] extends ScalarType[BaseType]

case object IntegerType extends NumericType[Int]

case object RealType extends NumericType[Double]

case object BooleanType extends ScalarType[Boolean]

case class ArrayType[ElementType](elemType : ElementType) extends ASTType[ElementType]

