package spn_compiler.backend.software.ast.nodes.statement

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}
import spn_compiler.backend.software.ast.construct.{ASTBuilder, _}
import spn_compiler.backend.software.ast.nodes.types.IntegerType

@RunWith(classOf[JUnitRunner])
class ASTBlockStatementTest extends FlatSpec with Matchers {

  private val builder = new ASTBuilder
  private val varFive = builder.createVariable(IntegerType, "five")
  private val stmt5 = builder.assignVariable(varFive, builder.constantValue(IntegerType, 5))
  private val varSix = builder.createVariable(IntegerType, "six")
  private val stmt6 = builder.assignVariable(varSix, builder.constantValue(IntegerType, 6))
  private val varSeven = builder.createVariable(IntegerType, "seven")
  private val stmt7 = builder.assignVariable(varSeven, builder.constantValue(IntegerType, 7))
  private val block = new ASTBlockStatement(builder)

  "A statement" should "be appendable to a block" in {
    val stmt = block ++ stmt6
    val ASTBlockStatement(testStmt) = block
    stmt should be(stmt6)
    testStmt should be(stmt6)
  }

  it should "be inserted before another statement in a block" in {
    val stmt = stmt6.insertBefore(stmt5)
    val ASTBlockStatement(s1, s2) = block
    stmt should be(stmt5)
    s1 should be(stmt5)
    s2 should be(stmt6)
  }

  it should "be inserted after another statement in a block" in {
    val stmt = stmt6.insertAfter(stmt7)
    val ASTBlockStatement(s1, s2, s3) = block
    stmt should be(stmt7)
    s1 should be(stmt5)
    s2 should be(stmt6)
    s3 should be(stmt7)
  }

  it should "be removable from a block" in {
    stmt6.delete()
    val ASTBlockStatement(s1, s2) = block
    s1 should be(stmt5)
    s2 should be(stmt7)
  }


}
