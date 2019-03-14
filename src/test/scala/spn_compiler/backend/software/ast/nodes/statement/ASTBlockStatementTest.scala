package spn_compiler.backend.software.ast.nodes.statement

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}
import spn_compiler.backend.software.ast.construct._
import spn_compiler.backend.software.ast.nodes.module.ASTModule
import spn_compiler.backend.software.ast.nodes.types.IntegerType

@RunWith(classOf[JUnitRunner])
class ASTBlockStatementTest extends FlatSpec with Matchers {

  private val builder = new ASTModule("test-dummy")
  private val block = new ASTBlockStatement
  builder.setInsertionPoint(block)
  private val varFive = builder.createVariable(IntegerType, "five")
  private val varSix = builder.createVariable(IntegerType, "six")
  private val varSeven = builder.createVariable(IntegerType, "seven")
  private var stmt5 : ASTStatement = _
  private var stmt6 : ASTStatement = _
  private var stmt7 : ASTStatement = _

  "A statement" should "be appendable to a block" in {
    stmt6 = builder.assignVariable(varSix, builder.constantValue(IntegerType, 6))
    val ASTBlockStatement(testStmt) = block
    testStmt should be(stmt6)
  }

  it should "be inserted before another statement in a block" in {
    builder.setInsertionPointBefore(stmt6)
    stmt5 = builder.assignVariable(varFive, builder.constantValue(IntegerType, 5))
    val ASTBlockStatement(s1, s2) = block
    s1 should be(stmt5)
    s2 should be(stmt6)
  }

  it should "be inserted after another statement in a block" in {
    builder.setInsertionPointAfter(stmt6)
    stmt7 = builder.assignVariable(varSeven, builder.constantValue(IntegerType, 7))
    val ASTBlockStatement(s1, s2, s3) = block
    s1 should be(stmt5)
    s2 should be(stmt6)
    s3 should be(stmt7)
  }

  it should "be removable from a block" in {
    builder.deleteStatement(stmt6)
    val ASTBlockStatement(s1, s2) = block
    s1 should be(stmt5)
    s2 should be(stmt7)
  }

  "It" should "be possible to insert a statement after deleting the prior insertion point statement" in {
    val stmt = builder.assignVariable(varFive, builder.constantValue(IntegerType, 42))
    val ASTBlockStatement(s1, s2, s3) = block
    s1 should be(stmt5)
    s2 should be(stmt7)
    s3 should be(stmt)
  }




}
