package spn_compiler.driver

import java.io.File

import scopt._
import spn_compiler.backend.software.codegen.CodeWriter
import spn_compiler.backend.software.codegen.cpp.{CppHeaderCodeGeneration, CppImplCodeGeneration}
import spn_compiler.backend.software.cpu.ast_generation.serial.SerialASTGeneration
import spn_compiler.driver.config.{BaseConfig, CLIConfig, CPPCompileConfig}
import spn_compiler.driver.option.{BaseOptions, CPPCompileOptions}
import spn_compiler.frontend.parser.Parser
import spn_compiler.util.logging.Logging
import spn_compiler.util.statistics.GraphStatistics

class CmdConfig extends CLIConfig[CmdConfig] with BaseConfig[CmdConfig] with CPPCompileConfig[CmdConfig] {
  override def self: CmdConfig = this
}

object Driver extends App with Logging {

  val builder = OParser.builder[CmdConfig]
  val cliParser : OParser[_, CmdConfig] = {
    import builder._
    OParser.sequence(
      programName("spnc"),
      head("spnc", "0.0.1"),
      BaseOptions.apply,
      CPPCompileOptions.apply
    )
  }

  val cliConfig : CmdConfig = OParser.parse(cliParser, args, new CmdConfig())
    .getOrElse(throw new RuntimeException("CLI Error!"))

   val spn = Parser.parseFile(cliConfig.in)

   if(cliConfig.computeStats){
     GraphStatistics.computeStatistics(spn, cliConfig.statsFile)
   }

   val ast = new SerialASTGeneration().createAST(spn)

   val headerFile = "spn.hpp"
   CppHeaderCodeGeneration(ast, CodeWriter(new File(headerFile))).generateHeader()
   new CppImplCodeGeneration(ast, headerFile, CodeWriter(new File("spn.cpp"))).generateCode()

}

