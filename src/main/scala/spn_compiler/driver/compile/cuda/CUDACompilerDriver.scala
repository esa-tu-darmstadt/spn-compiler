package spn_compiler.driver.compile.cuda

import spn_compiler.backend.software.codegen.CodeWriter
import spn_compiler.backend.software.codegen.cuda.{CUDAHeaderCodeGeneration, CUDAImplCodeGeneration}
import spn_compiler.backend.software.gpu.ast_generation.cuda.CUDAASTGeneration
import spn_compiler.driver.config.CompilerConfig
import spn_compiler.graph_ir.nodes.IRGraph
import spn_compiler.util.file.FileUtil
import spn_compiler.util.logging.Logging

object CUDACompilerDriver extends Logging {

  def execute[C <: CompilerConfig[C]](spn : IRGraph, config : C) : Unit = {
    trace("Creating AST for CUDA compilation...")
    val ast = new CUDAASTGeneration().createAST(spn)

    val codeDirectory = if(config.outputCodeOnly){
      info(s"Ignoring output file ${config.outputFile}, writing only code output")
      FileUtil.getParentDirectory(config.outputFile)
    } else FileUtil.getTmpDirectory

    val headerName = "spn.hpp"
    val headerFile = FileUtil.createFileInDirectory(codeDirectory, headerName)
    debug(s"Writing SPN CUDA header to ${headerFile.getAbsoluteFile.toString}")
    new CUDAHeaderCodeGeneration(ast, CodeWriter(headerFile)).generateHeader()
    val implFile = FileUtil.createFileInDirectory(codeDirectory, "spn.cu")
    debug(s"Writing SPN CUDA implementation to ${implFile.getAbsoluteFile.toString}")
    new CUDAImplCodeGeneration(ast, headerName, CodeWriter(implFile)).generateCode()
    val mainFile = FileUtil.createFileInDirectory(codeDirectory, "main.cpp")
    debug(s"Writing C++ main implementation to ${mainFile.getAbsoluteFile.toString}")
    CUDAEntryPoint.writeMain(mainFile)
    if(config.outputCodeOnly){
      return
    }
    NVCCCompilerDriver.compile(config, implFile, mainFile)
  }

}
