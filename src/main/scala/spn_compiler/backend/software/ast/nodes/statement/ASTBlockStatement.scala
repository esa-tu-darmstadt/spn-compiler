package spn_compiler.backend.software.ast.nodes.statement

import scala.collection.mutable.ListBuffer

class ASTBlockStatement private[ast] {

  private val statements : ListBuffer[ASTStatement] = ListBuffer()

  private[ast] def insertBefore[Stmt <: ASTStatement](insertionPoint: ASTStatement, stmt : Stmt) : Stmt = {
    require(statements.contains(insertionPoint), "Can only insert before statement already contained in this block!")
    stmt.setBlock(this)
    statements.insert(statements.indexOf(insertionPoint), stmt)
    stmt
  }

  private[ast] def append[Stmt <: ASTStatement](stmt : Stmt): Stmt = {
    stmt.setBlock(this)
    statements.append(stmt)
    stmt
  }

  private[ast] def delete(stmt : ASTStatement) : Unit = {
    if(statements.contains(stmt)){
      statements.remove(statements.indexOf(stmt))
    }
  }

  private[ast] def getNextStatement(stmt : ASTStatement) : Option[ASTStatement] = {
    require(statements.contains(stmt), "Statement is not part of this block")
    if(statements.last == stmt){
      None
    }
    else {
      Some(statements(statements.indexOf(stmt)+1))
    }
  }

}

object ASTBlockStatement {

  def unapplySeq(block : ASTBlockStatement) : Option[Seq[ASTStatement]] = Some(Seq(block.statements:_*))

}