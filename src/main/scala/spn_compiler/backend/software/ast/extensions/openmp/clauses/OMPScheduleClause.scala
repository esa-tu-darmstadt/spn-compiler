package spn_compiler.backend.software.ast.extensions.openmp.clauses

class OMPScheduleClause private[ast](val kind : OMPScheduleClause.Kind, val chunkSize : Option[Int] = None)
  extends OMPWorksharingLoopClause with OMPParallelWorksharingLoopClause

object OMPScheduleClause {

  sealed abstract class Kind(val syntax : String)
  case object STATIC extends Kind("static")
  case object DYNAMIC extends Kind("dynamic")
  case object GUIDED extends Kind("guided")

  def unapply(c : OMPScheduleClause) : Option[(Kind, Option[Int])] = Some(c.kind, c.chunkSize)

}
