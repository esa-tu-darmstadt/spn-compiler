package spn_compiler.backend.software.codegen.openmp

import spn_compiler.backend.software.ast.extensions.openmp.clauses.{OMPClause, OMPDataSharingClause, OMPScheduleClause}
import spn_compiler.backend.software.ast.extensions.openmp.constructs.OMPParallelWorksharingLoop
import spn_compiler.backend.software.ast.nodes.module.ASTModule
import spn_compiler.backend.software.ast.nodes.statement.ASTStatement
import spn_compiler.backend.software.codegen.CodeWriter
import spn_compiler.backend.software.codegen.cpp.CppImplCodeGeneration

class OMPImplCodeGeneration(ast : ASTModule, headerName : String,  writer : CodeWriter)
  extends CppImplCodeGeneration(ast, headerName, writer) {

  override def generateCode() : Unit = {
    val ASTModule(name, _, globalVars, functions) = ast
    writer.writeln("#include \"%s\"".format(headerName))
    writer.writeln("#include <omp.h>")
    globalVars.foreach(writeGlobalVariable)
    functions.foreach(writeFunction)
    writer.close()
  }

  override protected def writeStatement(stmt: ASTStatement) : Unit = stmt match {
    case OMPParallelWorksharingLoop(loop, clauses @ _*) => {
      writer.writeln("#pragma omp parallel for %s".format(clauses.map(writeClause).mkString(" ")))
      writeStatement(loop)
    }
    case _ => super.writeStatement(stmt)
  }

  protected def writeClause(clause : OMPClause) : String = clause match {
    case ds @ OMPDataSharingClause(variables @ _*) =>
      "%s(%s)".format(ds.syntax, variables.map(_.name).mkString(","))
    case OMPScheduleClause(kind, chunkSize) => {
      val suffix = if(chunkSize.isDefined) ",%d".format(chunkSize.get) else ""
      "schedule(%s%s)".format(kind.syntax, suffix)
    }
    case _ => throw new RuntimeException("Unknown OpenMP clause %s".format(clause.toString))
  }
}
