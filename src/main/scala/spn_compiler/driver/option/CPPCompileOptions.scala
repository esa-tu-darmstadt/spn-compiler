package spn_compiler.driver.option

import scopt._
import spn_compiler.driver.config.CPPCompileConfig

object CPPCompileOptions {

  def apply[R <: CPPCompileConfig[R]] : OParser[_, R] = {
    val builder = OParser.builder[R]
    val parser = {
      import builder._
      OParser.sequence(
        opt[Int]('O', "optimization-level")
          .action((l, c) => c.setOptimizationLevel(l))
          .text("Optimization level for C++ compilation"),
      )
    }
    parser
  }
}
