package spn_compiler.backend.software.ast.nodes.value.expression

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}
import spn_compiler.backend.software.ast.nodes.module.ASTModule
import spn_compiler.backend.software.ast.nodes.types.IntegerType

@RunWith(classOf[JUnitRunner])
class ASTArithmeticExpressionTest extends FlatSpec with Matchers {

  val builder = new ASTModule("test-dummy")
  val constantFive = builder.constantValue(IntegerType, 5)
  val constantSix = builder.constantValue(IntegerType, 6)
  val addFiveSix = builder.add(constantFive, constantSix)

  "An addition" should "be constructable" in {

    "val builder = new ASTModule(\"test-dummy\")\n" +
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

  "A compare" should "be constructable, contain correct operands and pattern-match" in {
    val cmp = builder.cmpLE(constantSix, constantFive)
    val ASTBinaryExpression(testLeft1, testRight1) = cmp
    testLeft1 should be(constantSix)
    testRight1 should be(constantFive)
    val ASTCompareExpression(testLeft2, testRight2) = cmp
    testLeft2 should be(constantSix)
    testRight2 should be(constantFive)
    val ASTCmpLE(testLeft3, testRight3) = cmp
    testLeft3 should be(constantSix)
    testRight3 should be(constantFive)
    intercept[MatchError]{val ASTCmpGE(l, r) = cmp}
  }

  "A not" should "be constructable, contain correct operand and pattern-match" in {
    val not = builder.not(constantFive)
    val ASTUnaryExpression(testOp1) = not
    testOp1 should be(constantFive)
    val ASTNot(testOp2) = not
    testOp2 should be(constantFive)
  }

}
