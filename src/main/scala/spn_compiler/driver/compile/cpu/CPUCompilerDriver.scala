package spn_compiler.driver.compile.cpu

import spn_compiler.backend.software.codegen.CodeWriter
import spn_compiler.backend.software.codegen.cpp.{CppHeaderCodeGeneration, CppImplCodeGeneration}
import spn_compiler.backend.software.codegen.openmp.OMPImplCodeGeneration
import spn_compiler.backend.software.cpu.ast_generation.openmp.OMPASTGeneration
import spn_compiler.backend.software.cpu.ast_generation.serial.ASTGeneration
import spn_compiler.driver.config.{CPPCompileConfig, CompilerConfig}
import spn_compiler.graph_ir.nodes.IRGraph
import spn_compiler.util.file.FileUtil
import spn_compiler.util.logging.Logging

object CPUCompilerDriver extends Logging {

  def execute[C <: CPPCompileConfig[C] with CompilerConfig[C]](spn : IRGraph, config : C): Unit = {
    trace(s"Creating AST for C++ compilation...")
    val astGenerator : ASTGeneration = if(config.isOMPParallelForEnabled){
      info("OpenMP parallel is enabled, using OpenMP worksharing loop for parallel processing")
      new OMPASTGeneration
    }
    else {
      new ASTGeneration
    }
    val ast = astGenerator.createAST(spn)
    val codeDirectory = if(config.outputCodeOnly){
      info(s"Ignoring output file ${config.outputFile}, writing only code output")
      FileUtil.getParentDirectory(config.outputFile)
    } else FileUtil.createScratchpadDirectory

    val headerName = "spn.hpp"
    val headerFile = FileUtil.createFileInDirectory(codeDirectory, headerName)
    debug(s"Writing SPN C++ header to ${headerFile.getAbsoluteFile.toString}")
    CppHeaderCodeGeneration(ast, CodeWriter(headerFile)).generateHeader()
    val implFile = FileUtil.createFileInDirectory(codeDirectory, "spn.cpp")
    debug(s"Writing SPN C++ implementation to ${implFile.getAbsoluteFile.toString}")
    if(config.isOMPParallelForEnabled){
      new OMPImplCodeGeneration(ast, headerName, CodeWriter(implFile)).generateCode()
    }
    else {
      new CppImplCodeGeneration(ast, headerName, CodeWriter(implFile)).generateCode()
    }
    val mainFile = FileUtil.createFileInDirectory(codeDirectory, "main.cpp")
    debug(s"Writing C++ main implementation to ${mainFile.getAbsoluteFile.toString}")
    CPPEntryPoint.writeMain(mainFile)
    if(config.outputCodeOnly){
      return
    }
    config.compilerDriver.compile(config, implFile, mainFile)
  }

}
