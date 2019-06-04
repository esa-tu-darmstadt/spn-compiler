package spn_compiler.driver.option

import java.io.File

import scopt._
import spn_compiler.driver.config.BaseConfig
import spn_compiler.util.logging.Logging.{VerbosityExtraVerbose, VerbosityVerbose}

object BaseOptions {

  def apply[R <: BaseConfig[R]] : OParser[_, R] = {
    val builder = OParser.builder[R]
    val parser = {
      import builder._
      OParser.sequence(
        help('h', "help").text("Print usage information"),
        arg[File]("<input-file>")
          .action((f,c) => c.setInputFile(f))
          .validate(f => if(f.isFile && f.exists) success else failure("Input file must be an existing file!"))
          .text("Input file")
          .required(),
        opt[Unit]('v', "verbose")
          .action((_, c) => c.setVerbosityLevel(VerbosityVerbose))
          .text("Use verbose output"),
        opt[Unit]("very-verbose")
          .action((_, c) => c.setVerbosityLevel(VerbosityExtraVerbose))
          .text("Use very verbose output"),
        opt[Unit]('s', "stats")
          .action((x, c) => c.enableStats(true))
          .text("Compute statistics about the SPN graph"),
        opt[File]("stats-out")
          .action((f, c) => c.setStatsFile(f))
          .text("Output file for statistics, default: stats.spns"),
        opt[String]('t', "target")
          .validate(target =>
            if(BaseConfig.availableTargets.exists(_.name.equalsIgnoreCase(target))){
              success
            }
            else {
              failure(s"No matching target with name $target")
            })
          .action((target, c) =>
            c.setTarget(BaseConfig.availableTargets.filter(_.name.equalsIgnoreCase(target)).head))
          .text(s"Compilation target. Available targets: ${BaseConfig.availableTargets.map(_.name).mkString(", ")}")
      )
    }
    parser
  }

}
