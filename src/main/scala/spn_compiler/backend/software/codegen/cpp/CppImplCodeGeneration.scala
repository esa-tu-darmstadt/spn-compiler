package spn_compiler.backend.software.codegen.cpp

import spn_compiler.backend.software.ast.nodes.function.ASTFunction
import spn_compiler.backend.software.ast.nodes.module.ASTModule
import spn_compiler.backend.software.ast.nodes.statement.control_flow.{ASTCallStatement, ASTForLoop, ASTIfStatement, ASTReturnStatement}
import spn_compiler.backend.software.ast.nodes.statement.variable.{ASTVariableAssignment, ASTVariableDeclaration}
import spn_compiler.backend.software.ast.nodes.statement.{ASTBlockStatement, ASTStatement}
import spn_compiler.backend.software.codegen.CodeWriter

class CppImplCodeGeneration(ast : ASTModule, headerName : String,  writer : CodeWriter) extends CppValueCodeGeneration
  with CppReferenceCodeGeneration with CppTypeCodeGeneration {

  def generateCode() : Unit = {
    val ASTModule(name, _, globalVars, functions) = ast
    writer.writeln("#include \"%s\"".format(headerName))
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
    writeBlockStatement(function.body)
  }

  protected def writeBlockStatement(block : ASTBlockStatement) : Unit = {
    writer.writeln("{")
    val ASTBlockStatement(statements @ _*) = block
    statements.foreach(writeStatement)
    writer.writeln("}")
  }

  protected def writeStatement(stmt : ASTStatement) : Unit = stmt match {
    case ASTIfStatement(testVal, thenBlock, elseBlock) => {
      writer.write("if(%s)".format(generateValue(testVal)))
      writeBlockStatement(thenBlock)
      writer.write("else")
      writeBlockStatement(elseBlock)
    }

    case ASTVariableAssignment(ref, value) =>
      writer.writeln("%s = %s;".format(generateReference(ref), generateValue(value)))

    case ASTVariableDeclaration(variable, initValue) => {
      val appendix = if(initValue.isDefined) " = %s".format(generateValue(initValue.get)) else ""
      writer.writeln("%s %s%s;".format(generateType(variable.ty), variable.name, appendix))
    }

    case ASTForLoop(initVar, initVal, testVal, incrVar, incrVal, block) => {
      // If initVar is defined, initVal must be defined, too. This is ensured by require in class ASTForLoop.
      val init = if(initVar.isDefined) "%s = %s".format(generateReference(initVar.get), generateValue(initVal.get)) else ""
      // If incrVar is defined, incrVal must be defined, too. This is ensured by require in class ASTForLoop.
      val incr = if(incrVar.isDefined) "%s = %s".format(generateReference(incrVar.get), generateValue(incrVal.get)) else ""
      writer.write("for(%s; %s; %s)".format(init, generateValue(testVal), incr))
      writeBlockStatement(block)
    }

    case ASTReturnStatement(value) => writer.writeln("return %s;".format(generateValue(value)))

    case call : ASTCallStatement => writer.writeln("%s;".format(generateValue(call.call)))
  }

}
