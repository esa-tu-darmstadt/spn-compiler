package spn_compiler.driver.option

import java.io.File

import scopt._
import spn_compiler.driver.config.BaseConfig

object BaseOptions {

  def apply[R <: BaseConfig[R]] : OParser[_, R] = {
    val builder = OParser.builder[R]
    val parser = {
      import builder._
      OParser.sequence(
        opt[File]("stats-out")
          .action((f, c) => c.setStatsFile(f))
          .text("Output file for statistics, default stats.spns"),
        opt[Unit]('s', "stats")
          .action((x, c) => c.enableStats(true)),
        arg[File]("<input-file>")
          .action((f,c) => c.setInputFile(f)),
        help('h', "help").text("Print usage information")
      )
    }
    parser
  }

}
