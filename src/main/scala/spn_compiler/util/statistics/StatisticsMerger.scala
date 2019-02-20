package spn_compiler.util.statistics

import java.io.{File, FileInputStream, FileOutputStream}
import java.nio.file.Files

import scopt.OParser

/**
  * The statistics merger tool allows to merge SPN graph statistics from
  * two different JSON files into a single one.
  * If the second input file does not exist, the first input is just copied
  * over to the output file.
  */
object StatisticsMerger extends App{

  val builder = OParser.builder[MergerCLIConfig]
  val cliParser = {
    import builder._
    OParser.sequence(
      programName("spn-stats-merge"),
      head("spn-stats-merge", "0.0.1"),
      arg[File]("<input 1>")
        .action((f, c) => c.copy(in1 = f))
        .required(),
      arg[File]("<input 2>")
        .action((f, c) => c.copy(in2 = f))
        .required(),
      arg[File]("<output>")
        .action((f, c) => c.copy(out = f))
        .required()
    )
  }

  val cliConfig : MergerCLIConfig = OParser.parse(cliParser, args, MergerCLIConfig())
    .getOrElse(throw new RuntimeException("CLI Error!"))

  if(!Files.exists(cliConfig.in2.toPath)){
    // Just copy the first input file to the output file.
    new FileOutputStream(cliConfig.out).getChannel
      .transferFrom(new FileInputStream(cliConfig.in1).getChannel, 0, Long.MaxValue)
  } else {
    GraphStatistics.mergeStatistics(cliConfig.in1, cliConfig.in2, cliConfig.out)
  }


}

case class MergerCLIConfig(in1 : File = new File("stats1.spns"),
                                   in2 : File = new File("stats2.spns"),
                                   out : File = new File("out.spns"))
