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
