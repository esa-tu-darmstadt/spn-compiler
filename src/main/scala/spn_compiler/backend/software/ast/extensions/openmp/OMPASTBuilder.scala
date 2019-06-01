package spn_compiler.backend.software.ast.extensions.openmp

import spn_compiler.backend.software.ast.construct.ASTBuilder
import spn_compiler.backend.software.ast.extensions.openmp.clauses._
import spn_compiler.backend.software.ast.extensions.openmp.constructs.{OMPCanonicalLoop, OMPConstruct, OMPParallelWorksharingLoop}
import spn_compiler.backend.software.ast.nodes.reference.ASTReferencable

trait OMPASTBuilder extends ASTBuilder {

  def ompParallelFor(loop : OMPCanonicalLoop) : OMPParallelWorksharingLoop = new OMPParallelWorksharingLoop(loop)

  def addClause[Clause <: OMPClause](construct : OMPConstruct[Clause], clause : Clause) : Unit = construct.addClause(clause)

  def shared(variables : ASTReferencable*) : OMPSharedClause = new OMPSharedClause(variables:_*)

  def privateClause(variables : ASTReferencable*) : OMPPrivateClause = new OMPPrivateClause(variables:_*)

  def firstPrivate(variables : ASTReferencable*) : OMPFirstPrivateClause = new OMPFirstPrivateClause(variables:_*)

  def lastPrivate(variables : ASTReferencable*) : OMPLastPrivateClause = new OMPLastPrivateClause(variables:_*)

  def schedule(kind : OMPScheduleClause.Kind, chunkSize : Option[Int]) : OMPScheduleClause =
    new OMPScheduleClause(kind, chunkSize)

}
