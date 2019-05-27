package spn_compiler.backend.software.codegen.cpp

import spn_compiler.backend.software.ast.nodes.function.ASTFunction
import spn_compiler.backend.software.ast.nodes.module.ASTModule
import spn_compiler.backend.software.ast.nodes.statement.variable.ASTVariableDeclaration
import spn_compiler.backend.software.codegen.CodeWriter

class CppImplCodeGeneration(ast : ASTModule, headerName : String, val writer : CodeWriter) extends CppValueCodeGeneration
  with CppReferenceCodeGeneration with CppTypeCodeGeneration with CppStatementCodeGeneration {

  def generateCode() : Unit = {
    val ASTModule(name, headers, _, globalVars, functions) = ast
    writer.writeln("#include \"%s\"".format(headerName))
    headers.foreach(h => writer.writeln("#include <%s>".format(h)))
    globalVars.foreach(writeGlobalVariable)
    functions.foreach(writeFunction)
    writer.close()
  }

  protected def writeGlobalVariable(decl : ASTVariableDeclaration) : Unit = {
    val appendix = if(decl.initValue.isDefined) " = %s".format(generateValue(decl.initValue.get)) else ""
    writer.writeln("%s%s;".format(declareVariable(decl.variable.ty, decl.variable.name), appendix))
  }

  protected def writeFunction(function : ASTFunction) : Unit = {
    val parameters = function.getParameters.map(p => "%s %s".format(generateType(p.ty), p.name))
    writer.write("%s %s(%s)".format(generateType(function.returnType),
      function.name, parameters.mkString(",")))
    generateBlockStatement(function.body)
  }

}
