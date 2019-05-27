package spn_compiler.backend.software.codegen.cuda

import spn_compiler.backend.software.ast.extensions.cuda.statement.{CUDADim3Init, CUDAKernelInvocation}
import spn_compiler.backend.software.ast.nodes.statement.ASTStatement
import spn_compiler.backend.software.codegen.cpp.CppStatementCodeGeneration

trait CUDAStatementCodeGeneration extends CppStatementCodeGeneration {

  override def generateStatement(stmt: ASTStatement): Unit = stmt match {
    case CUDADim3Init(variable, x, y, z) =>
      writer.writeln("dim3 %s(%s, %s, %s);".format(variable.name, generateValue(x), generateValue(y), generateValue(z)))

    case CUDAKernelInvocation(gridLayout, blockLayout, kernel, params @ _*) => {
      val parameters = params.map(generateValue).mkString(",")
      writer.writeln("%s<<<%s,%s>>>(%s);".format(kernel.name, generateValue(gridLayout),
        generateValue(blockLayout), parameters))
    }
  }

}
