package spn_compiler.backend.software.codegen.openmp

import spn_compiler.backend.software.ast.extensions.openmp.clauses.{OMPClause, OMPDataSharingClause, OMPScheduleClause}
import spn_compiler.backend.software.ast.extensions.openmp.constructs.OMPParallelWorksharingLoop
import spn_compiler.backend.software.ast.nodes.statement.ASTStatement
import spn_compiler.backend.software.codegen.cpp.CppStatementCodeGeneration

trait OMPConstructCodeGeneration extends CppStatementCodeGeneration {

  override def generateStatement(stmt: ASTStatement) : Unit = stmt match {
    case OMPParallelWorksharingLoop(loop, clauses @ _*) => {
      writer.writeln("#pragma omp parallel for %s".format(clauses.map(generateClause).mkString(" ")))
      generateStatement(loop)
    }
    case _ => super.generateStatement(stmt)
  }

  protected def generateClause(clause : OMPClause) : String = clause match {
    case ds @ OMPDataSharingClause(variables @ _*) =>
      "%s(%s)".format(ds.syntax, variables.map(_.name).mkString(","))
    case OMPScheduleClause(kind, chunkSize) => {
      val suffix = if(chunkSize.isDefined) ",%d".format(chunkSize.get) else ""
      "schedule(%s%s)".format(kind.syntax, suffix)
    }
    case _ => throw new RuntimeException("Unknown OpenMP clause %s".format(clause.toString))
  }
}
