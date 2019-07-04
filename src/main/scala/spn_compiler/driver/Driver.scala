package spn_compiler.driver

import scopt._
import spn_compiler.driver.compile.cpu.CPUCompilerDriver
import spn_compiler.driver.compile.cuda.CUDACompilerDriver
import spn_compiler.driver.config._
import spn_compiler.driver.option.{BaseOptions, CPPCompileOptions, CUDACompileOptions, CompilerOptions}
import spn_compiler.frontend.parser.Parser
import spn_compiler.util.logging.Logging
import spn_compiler.util.statistics.GraphStatistics

class DriverConfig extends CLIConfig[DriverConfig]
  with BaseConfig[DriverConfig]
  with CompilerConfig[DriverConfig]
  with CPPCompileConfig[DriverConfig]
  with CUDACompileConfig[DriverConfig]{
  override def self: DriverConfig = this
}

object Driver extends App with Logging {

  val builder = OParser.builder[DriverConfig]
  val cliParser : OParser[_, DriverConfig] = {
    import builder._
    OParser.sequence(
      programName("spnc"),
      head("spnc", "0.0.3"),
      BaseOptions.apply,
      CompilerOptions.apply,
      CPPCompileOptions.apply,
      CUDACompileOptions.apply
    )
  }

  val cliConfig : DriverConfig = OParser.parse(cliParser, args, new DriverConfig())
    .getOrElse(throw new RuntimeException("CLI Error!"))

  Logging.setVerbosityLevel(cliConfig.verbosityLevel)

  val (spn, marginals) = Parser.parseFile(cliConfig.in)

  if(cliConfig.computeStats){
   GraphStatistics.computeStatistics(spn, cliConfig.statsFile)
  }

  cliConfig.target match {
    case BaseConfig.CPPTarget => CPUCompilerDriver.execute(spn, cliConfig)
    case BaseConfig.CUDATarget => CUDACompilerDriver.execute(spn, cliConfig)
  }


}

