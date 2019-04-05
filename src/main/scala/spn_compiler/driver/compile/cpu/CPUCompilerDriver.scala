package spn_compiler.driver.compile.cpu

import spn_compiler.backend.software.codegen.CodeWriter
import spn_compiler.backend.software.codegen.cpp.{CppHeaderCodeGeneration, CppImplCodeGeneration}
import spn_compiler.backend.software.cpu.ast_generation.serial.SerialASTGeneration
import spn_compiler.driver.config.CPPCompileConfig
import spn_compiler.graph_ir.nodes.IRGraph
import spn_compiler.util.file.FileUtil
import spn_compiler.util.logging.Logging

object CPUCompilerDriver extends Logging {

  def execute[C <: CPPCompileConfig[C]](spn : IRGraph, config : C): Unit = {
    trace("Creating AST for C++ compilation...")
    val ast = new SerialASTGeneration().createAST(spn)
    val codeDirectory = if(config.outputCodeOnly){
      info(s"Ignoring output file ${config.outputFile}, writing only code output")
      FileUtil.getParentDirectory(config.outputFile)
    } else FileUtil.getTmpDirectory

    val headerName = "spn.hpp"
    val headerFile = FileUtil.createFileInDirectory(codeDirectory, headerName)
    debug(s"Writing SPN C++ header to ${headerFile.getAbsoluteFile.toString}")
    CppHeaderCodeGeneration(ast, CodeWriter(headerFile)).generateHeader()
    val implFile = FileUtil.createFileInDirectory(codeDirectory, "spn.cpp")
    debug(s"Writing SPN C++ implementation to ${implFile.getAbsoluteFile.toString}")
    new CppImplCodeGeneration(ast, headerName, CodeWriter(implFile)).generateCode()
    val mainFile = FileUtil.createFileInDirectory(codeDirectory, "main.cpp")
    debug(s"Writing C++ main implementation to ${implFile.getAbsoluteFile.toString}")
    CPPEntryPoint.writeMain(mainFile)
    if(config.outputCodeOnly){
      return
    }
    config.compilerDriver.compile(config, implFile, mainFile)
  }

}
