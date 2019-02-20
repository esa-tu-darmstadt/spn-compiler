package spn_compiler.driver

import java.io.File

import scopt._
import spn_compiler.frontend.parser.Parser
import spn_compiler.util.statistics.GraphStatistics

object Driver extends App {

  val builder = OParser.builder[CLIConfig]
  val cliParser = {
    import builder._
    OParser.sequence(
      programName("spnc"),
      head("spnc", "0.0.1"),
      opt[File]("stats-out")
        .action((x, c) => c.copy(statsFile = x))
        .text("Output file for statistics, default stats.spns"),
      opt[Unit]('s', "stats")
        .action((x, c) => c.copy(computeStats = true)),
      arg[File]("<input-file>")
        .action((f,c) => c.copy(in = f))
    )
  }

  val cliConfig : CLIConfig = OParser.parse(cliParser, args, CLIConfig())
    .getOrElse(throw new RuntimeException("CLI Error!"))

  val spn = Parser.parseFile(cliConfig.in)

  if(cliConfig.computeStats){
    GraphStatistics.computeStatistics(spn, cliConfig.statsFile)
  }
}

case class CLIConfig(in : File = new File("structure.spn"), statsFile : File = new File("stats.spns"),
                     computeStats : Boolean = false)
