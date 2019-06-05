package spn_compiler.backend.software.codegen.cuda

import spn_compiler.backend.software.ast.extensions.cuda.CUDAModule
import spn_compiler.backend.software.ast.extensions.cuda.function.CUDAFunction
import spn_compiler.backend.software.ast.extensions.cuda.statement.{CUDADim3Init, CUDAKernelInvocation}
import spn_compiler.backend.software.ast.nodes.function.ASTFunction
import spn_compiler.backend.software.ast.nodes.statement.ASTStatement
import spn_compiler.backend.software.codegen.CodeWriter
import spn_compiler.backend.software.codegen.cpp.CppImplCodeGeneration

class CUDAImplCodeGeneration(ast : CUDAModule, headerName : String, writer : CodeWriter)
  extends CppImplCodeGeneration(ast, headerName, writer) {

  override protected def writeFunction(function: ASTFunction): Unit = function match {
    case kernel @ CUDAFunction(scope, name, returnType, parameters @ _*) => {
      writer.writeln("%s %s %s(%s)".format(scope.prefix, generateType(returnType), name,
        parameters.map(p => "%s %s".format(generateType(p.ty), p.name)).mkString(",")))
      generateBlockStatement(kernel.body)
    }
    case _ => super.writeFunction(function)
  }

  override def generateStatement(stmt: ASTStatement): Unit = stmt match {
    case CUDADim3Init(variable, x, y, z) =>
      writer.writeln("dim3 %s(%s, %s, %s);".format(variable.name, generateValue(x), generateValue(y), generateValue(z)))

    case CUDAKernelInvocation(gridLayout, blockLayout, kernel, params @ _*) => {
      val parameters = params.map(generateValue).mkString(",")
      writer.writeln("%s<<<%s,%s>>>(%s);".format(kernel.name, generateValue(gridLayout),
        generateValue(blockLayout), parameters))
    }

    case _ => super.generateStatement(stmt)
  }

}
