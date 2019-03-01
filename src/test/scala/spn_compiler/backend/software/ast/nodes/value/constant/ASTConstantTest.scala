package spn_compiler.backend.software.ast.nodes.value.constant

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}
import spn_compiler.backend.software.ast.construct.ASTBuilder
import spn_compiler.backend.software.ast.nodes.types.IntegerType

@RunWith(classOf[JUnitRunner])
class ASTConstantTest extends FlatSpec with Matchers {

  val builder = new ASTBuilder()

  "A constant" should "be constructable" in {
    "val builder = new ASTBuilder()\n"+
    "val constantFive = builder.constantValue(IntegerType, 5)" should compile
  }

  it should "pattern-match" in {
    val constantFive = builder.constantValue(IntegerType, 5)
    val ASTConstant(IntegerType, five) = constantFive
    five should be(5)
  }
}
