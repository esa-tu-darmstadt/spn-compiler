package spn_compiler.backend.software.ast.nodes.value.expression

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}
import spn_compiler.backend.software.ast.construct.ASTBuilder
import spn_compiler.backend.software.ast.nodes.types.IntegerType

@RunWith(classOf[JUnitRunner])
class ASTBinaryExpressionTest extends FlatSpec with Matchers {

  val builder = new ASTBuilder
  val constantFive = builder.constantValue(IntegerType, 5)
  val constantSix = builder.constantValue(IntegerType, 6)
  val addFiveSix = builder.add(constantFive, constantSix)

  "An addition" should "be constructable" in {

    "val builder = new ASTBuilder\n" +
    "  val constantFive = builder.constantValue(IntegerType, 5)\n" +
    "  val constantSix = builder.constantValue(IntegerType, 6)\n" +
    "  val addFiveSix = builder.add(constantFive, constantSix)" should compile
  }

  it should "pattern-match and contain correct operands" in {
    val ASTAddition(leftOp, rightOp) = addFiveSix
    leftOp should be(constantFive)
    rightOp should be(constantSix)
  }

  it should "pattern-match a binary operator" in {
    val ASTBinaryExpression(leftOp, rightOp) = addFiveSix
    leftOp should be(constantFive)
    rightOp should be(constantSix)
  }

}
