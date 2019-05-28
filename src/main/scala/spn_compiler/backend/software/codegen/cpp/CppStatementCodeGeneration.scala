package spn_compiler.backend.software.codegen.cpp

import spn_compiler.backend.software.ast.nodes.statement.control_flow.{ASTCallStatement, ASTForLoop, ASTIfStatement, ASTReturnStatement}
import spn_compiler.backend.software.ast.nodes.statement.variable.{ASTVariableAssignment, ASTVariableDeclaration}
import spn_compiler.backend.software.ast.nodes.statement.{ASTBlockStatement, ASTStatement}
import spn_compiler.backend.software.codegen.{ReferenceCodeGeneration, StatementCodeGeneration, TypeCodeGeneration, ValueCodeGeneration}

trait CppStatementCodeGeneration extends StatementCodeGeneration
  with ValueCodeGeneration with ReferenceCodeGeneration with TypeCodeGeneration {

  def generateBlockStatement(block : ASTBlockStatement) : Unit = {
    writer.writeln("{")
    val ASTBlockStatement(statements @ _*) = block
    statements.foreach(generateStatement)
    writer.writeln("}")
  }

  def generateStatement(stmt : ASTStatement) : Unit = stmt match {
    case ASTIfStatement(testVal, thenBlock, elseBlock) => {
      writer.write("if(%s)".format(generateValue(testVal)))
      generateBlockStatement(thenBlock)
      if(!elseBlock.isEmpty){
        writer.write("else")
        generateBlockStatement(elseBlock)
      }
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
      generateBlockStatement(block)
    }

    case ASTReturnStatement(value) => writer.writeln("return %s;".format(generateValue(value)))

    case call : ASTCallStatement => writer.writeln("%s;".format(generateValue(call.call)))
  }

}
