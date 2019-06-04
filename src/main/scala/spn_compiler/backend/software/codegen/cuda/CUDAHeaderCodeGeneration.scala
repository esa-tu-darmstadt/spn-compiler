package spn_compiler.backend.software.codegen.cuda

import spn_compiler.backend.software.ast.extensions.cuda.CUDAModule
import spn_compiler.backend.software.ast.extensions.cuda.function.CUDAFunction
import spn_compiler.backend.software.ast.nodes.function.ASTFunctionPrototype
import spn_compiler.backend.software.codegen.CodeWriter
import spn_compiler.backend.software.codegen.cpp.CppHeaderCodeGeneration

class CUDAHeaderCodeGeneration(ast : CUDAModule, writer : CodeWriter) extends CppHeaderCodeGeneration(ast, writer) {

  override protected def writeFunction(function: ASTFunctionPrototype): Unit = function match {
    case CUDAFunction(scope, name, returnType, parameterTypes @ _*) =>
      writer.writeln("%s %s %s(%s);".format(scope.prefix, generateType(returnType), name,
        parameterTypes.map(p => generateType(p.ty)).mkString(",")))

    case _ => super.writeFunction(function)
  }

}
