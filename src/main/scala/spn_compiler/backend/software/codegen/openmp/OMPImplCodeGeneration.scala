package spn_compiler.backend.software.codegen.openmp

import spn_compiler.backend.software.ast.nodes.module.ASTModule
import spn_compiler.backend.software.codegen.CodeWriter
import spn_compiler.backend.software.codegen.cpp.CppImplCodeGeneration

class OMPImplCodeGeneration(ast : ASTModule, headerName : String,  writer : CodeWriter)
  extends CppImplCodeGeneration(ast, headerName, writer) with OMPConstructCodeGeneration {

  override def generateCode() : Unit = {
    val ASTModule(name, headers, _, globalVars, functions) = ast
    writer.writeln("#include \"%s\"".format(headerName))
    writer.writeln("#include <omp.h>")
    headers.foreach(h => writer.writeln("#include <%s>".format(h)))
    globalVars.foreach(writeGlobalVariable)
    functions.foreach(writeFunction)
    writer.close()
  }
}
