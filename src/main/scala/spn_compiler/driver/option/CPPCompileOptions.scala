package spn_compiler.driver.option

import scopt._
import spn_compiler.driver.config.CPPCompileConfig

object CPPCompileOptions {

  def apply[R <: CPPCompileConfig[R]] : OParser[_, R] = {
    val builder = OParser.builder[R]
    val parser = {
      import builder._
      OParser.sequence(
        opt[Unit]("openmp-parallel")
          .action((_, c) => c.enableOMPParallelFor(true))
          .text("Use OpenMP to parallelize processing of examples"),
        opt[Unit]("range-profiling")
          .action((_, c) => c.enableRangeProfiling(true))
          .text("Record all intermediate values in the tree and determine min and max values"),
        opt[Unit]("lns-sim")
            .action((_, c) => c.enableLNSSimulation(true))
            .text("Simulate LNS-based arithmetic in software"),
        opt[Int]("lns-int-bits")
            .action((b, c) => c.setLNSIntBits(b))
            .text("Set number of integer bits for the LNS fixed point format"),
        opt[Int]("lns-fraction-bits")
            .action((b,c) => c.setLNSFractionBits(b))
            .text("Set number of fraction bits for the LNS fixed point format"),
        opt[Double]("lns-interpolation-error")
            .action((e,c) => c.setLNSInterpolationError(e))
            .text("Set interpolation error for helper function in LNS addition"),
        opt[String]("cpp-compiler")
          .action((name, c) =>
            c.setCompiler(CPPCompileConfig.availableCompilers.filter(d => d._1.equalsIgnoreCase(name)).head._2))
          .validate(name =>
            if(CPPCompileConfig.availableCompilers.exists(d => d._1.equalsIgnoreCase(name))){
              success
            }
            else{
              failure(s"No matching C++ compiler with name $name")
            })
          .text(s"C++ compiler to use for compilation of CPU executable. " +
            s"Available compilers: ${CPPCompileConfig.availableCompilers.map(_._1).mkString(", ")}")
      )
    }
    parser
  }
}
