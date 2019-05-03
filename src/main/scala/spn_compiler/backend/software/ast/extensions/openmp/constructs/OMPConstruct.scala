package spn_compiler.backend.software.ast.extensions.openmp.constructs

import spn_compiler.backend.software.ast.extensions.openmp.clauses.OMPClause
import spn_compiler.backend.software.ast.nodes.statement.ASTStatement

import scala.collection.mutable.ListBuffer

abstract class OMPConstruct[Clause <: OMPClause] extends ASTStatement {

  protected val clauses : ListBuffer[Clause] = ListBuffer()

  private[openmp] def addClause(clause : Clause) : Unit = clauses.append(clause)

  def getClauses : List[Clause] = clauses.toList

}
