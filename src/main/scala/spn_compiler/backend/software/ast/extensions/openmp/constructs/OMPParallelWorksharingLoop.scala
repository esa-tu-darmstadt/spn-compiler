package spn_compiler.backend.software.ast.extensions.openmp.constructs

import spn_compiler.backend.software.ast.extensions.openmp.clauses.OMPParallelWorksharingLoopClause
import spn_compiler.backend.software.ast.nodes.statement.ASTStatement

class OMPParallelWorksharingLoop private[ast](val loop : OMPCanonicalLoop)
  extends OMPConstruct[OMPParallelWorksharingLoopClause] with OMPCanonicalLoop

object OMPParallelWorksharingLoop {

  def unapplySeq(l : OMPParallelWorksharingLoop) : Option[(OMPCanonicalLoop, Seq[OMPParallelWorksharingLoopClause])] =
    Some((l.loop, l.clauses))

}

trait OMPCanonicalLoop extends ASTStatement