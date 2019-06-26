package spn_compiler.driver.option

import scopt.OParser
import spn_compiler.driver.config.CUDACompileConfig

object CUDACompileOptions {

  def apply[R <: CUDACompileConfig[R]] : OParser[_, R] = {
    val builder = OParser.builder[R]
    val parser = {
      import builder._
      OParser.sequence(
        opt[Seq[Int]]("cuda-architectures")
          .valueName("<arch1>,<arch2>,...")
          .action((a, c) => c.setCUDAArchitectures(a))
          .text("CUDA architectures to generate PTX and CUBIN for, given as integer, e.g. 62 for compute_62/sm_62")
      )
    }
    parser
  }

}
