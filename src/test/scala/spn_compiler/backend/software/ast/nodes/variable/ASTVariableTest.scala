package spn_compiler.backend.software.ast.nodes.variable

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}
import spn_compiler.backend.software.ast.construct.ASTBuilder
import spn_compiler.backend.software.ast.nodes.types.IntegerType

@RunWith(classOf[JUnitRunner])
class ASTVariableTest extends FlatSpec with Matchers {

  val builder = new ASTBuilder
  val variable = builder.createVariable(IntegerType, "var")

  "A variable" should "be creatable through the ASTBuilder interface" in {
    "val builder = new ASTBuilder\n" +
      "  val variable = builder.createVariable(IntegerType, \"var\")" should compile
  }

  it should "pattern-match" in {
    val ASTVariable(IntegerType, name) = variable
    name should be("var")
  }

  "Variable names" should "be unique" in {
    val variable2 = builder.createVariable(IntegerType, "var")
    val ASTVariable(IntegerType, name) = variable2
    name should not be "var"
    val variable3 = builder.createVariable(IntegerType, "var_101")
    val variable4 = builder.createVariable(IntegerType, "var")
    val ASTVariable(IntegerType, name2) = variable4
    name2 should not be "var_101"
  }
}
