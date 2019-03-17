package spn_compiler.backend.software.codegen.cpp

import spn_compiler.backend.software.ast.nodes.function.ASTFunctionPrototype
import spn_compiler.backend.software.ast.nodes.module.ASTModule
import spn_compiler.backend.software.ast.nodes.statement.variable.ASTVariableDeclaration
import spn_compiler.backend.software.ast.nodes.types.StructType
import spn_compiler.backend.software.codegen.CodeWriter

case class CppHeaderCodeGeneration(ast : ASTModule, writer : CodeWriter) extends CppValueCodeGeneration
  with CppTypeCodeGeneration with CppReferenceCodeGeneration {

  def generateHeader() : Unit = {
    val ASTModule(_, structTypes, globalVars, functions) = ast
    structTypes.foreach(writeStructType)
    globalVars.foreach(writeGlobalVariable)
    functions.foreach(writeFunction)
  }

  protected def writeStructType(structType : StructType) : Unit = {
    val elements = structType.elements.
      map(e => "%s %s".format(generateType(e._2), e._1)).mkString("", ";", ";")
    writer.writeln("struct %s{%s};".format(structType.name, elements))
    writer.writeln("typedef struct %s %s;".format(structType.name, generateType(structType)))
  }

  protected def writeGlobalVariable(decl : ASTVariableDeclaration) : Unit = {
    val appendix = if(decl.initValue.isDefined) " = %s".format(generateValue(decl.initValue.get)) else ""
    writer.writeln("%s %s%s;".format(generateType(decl.variable.ty), decl.variable.name, appendix))
  }

  protected def writeFunction(function : ASTFunctionPrototype) : Unit = {
    writer.writeln("%s %s(%s);".format(generateType(function.returnType),
      function.name, function.getParameterTypes.map(generateType).mkString(",")))
  }

}
