package spn_compiler.driver.option

import java.io.File

import scopt.OParser
import spn_compiler.driver.config.CompilerConfig

object CompilerOptions {

  def apply[R <: CompilerConfig[R]] : OParser[_, R] = {
    val builder = OParser.builder[R]
    val parser = {
      import builder._
      OParser.sequence(
        opt[File]('o', "output")
          .action((f,c) => c.setOutFile(f))
          .validate(f => if(f.isFile) success else failure("Output file must be a file!"))
          .text("Output file, default: spn.out"),
        opt[Int]('O', "optimization-level")
          .action((l, c) => c.setOptimizationLevel(l))
          .validate(l =>
            if(l >= 0 && l < 4) success else failure("Optimization level must be in the range [0..3]!"))
          .text("Optimization level for C++ compilation"),
        opt[Unit]("fast-math")
          .action((_, c) => c.enableFastMath(true))
          .text("Allow aggressive, lossy floating-point optimizations"),
        opt[Unit]('S', "code-only")
          .action((_, c) => c.setCodeOnly(true))
          .text("Only write code output")
      )
    }
    parser
  }

}
