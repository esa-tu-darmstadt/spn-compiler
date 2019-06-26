package spn_compiler.backend.software.ast.extensions.openmp.clauses

import spn_compiler.backend.software.ast.nodes.reference.ASTReferencable

abstract class OMPDataSharingClause(val syntax : String, val variables : ASTReferencable*) extends OMPParallelClause
  with OMPWorksharingLoopClause with OMPParallelWorksharingLoopClause

object OMPDataSharingClause {

  def unapplySeq(c : OMPDataSharingClause) : Option[Seq[ASTReferencable]] = Some(c.variables)

}

class OMPSharedClause private[openmp](variables : ASTReferencable*) extends OMPDataSharingClause("shared", variables:_*) {

  def merge(other : OMPSharedClause) : OMPSharedClause = new OMPSharedClause(variables.toList:::other.variables.toList:_*)

}

object OMPSharedClause {

  def unapplySeq(s : OMPSharedClause) : Option[Seq[ASTReferencable]] = Some(s.variables)

}

class OMPPrivateClause private[openmp](variables : ASTReferencable*) extends OMPDataSharingClause("private", variables:_*) {

  def merge(other : OMPPrivateClause) : OMPPrivateClause = new OMPPrivateClause(variables.toList:::other.variables.toList:_*)

}

object OMPPrivateClause {

  def unapplySeq(s : OMPPrivateClause) : Option[Seq[ASTReferencable]] = Some(s.variables)

}

class OMPFirstPrivateClause private[openmp](variables : ASTReferencable*) extends OMPDataSharingClause("firstprivate", variables:_*) {

  def merge(other : OMPFirstPrivateClause) : OMPFirstPrivateClause = new OMPFirstPrivateClause(variables.toList:::other.variables.toList:_*)

}

object OMPFirstPrivateClause {

  def unapplySeq(s : OMPFirstPrivateClause) : Option[Seq[ASTReferencable]] = Some(s.variables)

}

class OMPLastPrivateClause private[openmp](variables : ASTReferencable*) extends OMPDataSharingClause("lastprivate", variables:_*) {

  def merge(other : OMPLastPrivateClause) : OMPLastPrivateClause = new OMPLastPrivateClause(variables.toList:::other.variables.toList:_*)

}

object OMPLastPrivateClause {

  def unapplySeq(s : OMPLastPrivateClause) : Option[Seq[ASTReferencable]] = Some(s.variables)

}
