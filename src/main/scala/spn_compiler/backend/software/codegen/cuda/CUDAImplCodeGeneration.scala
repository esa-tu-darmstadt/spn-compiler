package spn_compiler.backend.software.codegen.cuda

import spn_compiler.backend.software.ast.extensions.cuda.CUDAModule
import spn_compiler.backend.software.ast.extensions.cuda.function.CUDAFunction
import spn_compiler.backend.software.ast.nodes.function.ASTFunction
import spn_compiler.backend.software.codegen.CodeWriter
import spn_compiler.backend.software.codegen.cpp.CppImplCodeGeneration

class CUDAImplCodeGeneration(ast : CUDAModule, headerName : String, writer : CodeWriter)
  extends CppImplCodeGeneration(ast, headerName, writer) with CUDAStatementCodeGeneration {

  override protected def writeFunction(function: ASTFunction): Unit = function match {
    case kernel @ CUDAFunction(scope, name, returnType, parameterTypes @ _*) => {
      writer.writeln("%s %s %s(%s);".format(scope.prefix, generateType(returnType), name,
        parameterTypes.map(p => generateType(p.ty)).mkString(",")))
      generateBlockStatement(kernel.body)
    }
    case _ => super.writeFunction(function)
  }

}
