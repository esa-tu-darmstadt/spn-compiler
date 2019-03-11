package spn_compiler.backend.software.ast.nodes.statement

import spn_compiler.backend.software.ast.construct.ASTBuilder

import scala.collection.mutable.ListBuffer

class ASTBlockStatement private[ast](private val builder : ASTBuilder) {

  private val statements : ListBuffer[ASTStatement] = ListBuffer()

  def insertBefore(insertionPoint : ASTStatement, stmt : ASTStatement) : ASTStatement = {
    builder.insertBeforeInBlock(this, insertionPoint, stmt)
  }

  def insertAfter(insertionPoint : ASTStatement, stmt : ASTStatement) : ASTStatement = {
    builder.insertAfterInBlock(this, insertionPoint, stmt)
  }

  def append(stmt : ASTStatement) : ASTStatement = {
    builder.appendToBlock(this, stmt)
  }

  def ++(stmt : ASTStatement) : ASTStatement = append(stmt)

  def delete(stmt : ASTStatement) : Unit = {
    builder.deleteFromBlock(this, stmt)
  }

  private[ast] def addBefore(insertionPoint: ASTStatement, stmt : ASTStatement) : ASTStatement = {
    require(statements.contains(insertionPoint), "Can only insert before statement already contained in this block!")
    stmt.setBlock(this)
    statements.insert(statements.indexOf(insertionPoint), stmt)
    stmt
  }

  private[ast] def addAfter(insertionPoint: ASTStatement, stmt : ASTStatement) : ASTStatement = {
    require(statements.contains(insertionPoint), "Can only insert after statement already contained in this block!")
    stmt.setBlock(this)
    statements.insert(statements.indexOf(insertionPoint)+1, stmt)
    stmt
  }

  private[ast] def addAtEnd(stmt : ASTStatement): ASTStatement = {
    stmt.setBlock(this)
    statements.append(stmt)
    stmt
  }

  private[ast] def remove(stmt : ASTStatement) : Unit = {
    if(statements.contains(stmt)){
      statements.remove(statements.indexOf(stmt))
    }
  }

}

object ASTBlockStatement {

  def unapplySeq(block : ASTBlockStatement) : Option[Seq[ASTStatement]] = Some(Seq(block.statements:_*))

}