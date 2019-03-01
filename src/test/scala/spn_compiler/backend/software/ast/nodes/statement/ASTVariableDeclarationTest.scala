package spn_compiler.backend.software.ast.nodes.statement

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}
import spn_compiler.backend.software.ast.construct.ASTBuilder
import spn_compiler.backend.software.ast.nodes.types.IntegerType
import spn_compiler.backend.software.ast.nodes.variable.ASTVariable

@RunWith(classOf[JUnitRunner])
class ASTVariableDeclarationTest extends FlatSpec with Matchers{

  val builder = new ASTBuilder

  "A variable-declaration" should "be constructable through the ASTBuilder interface" in {
    "val builder = new ASTBuilder\n" +
    "val variable = builder.createVariable(IntegerType, \"var\")\n" +
    "val declaration = builder.declareVariable(variable)" should compile
  }

  it should "pattern-match" in {
    val variable = builder.createVariable(IntegerType, "var")
    val declaration = builder.declareVariable(variable)
    val ASTVariableDeclaration(variable2) = declaration
    variable2 should be(variable)
  }

  it should "not be constructable with an unknown variable" in {
    val variable2 = new ASTVariable(IntegerType, "var2")
    an [RuntimeException] should be thrownBy builder.declareVariable(variable2)
  }
}
