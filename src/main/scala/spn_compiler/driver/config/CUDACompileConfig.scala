package spn_compiler.driver.config

trait CUDACompileConfig[R <: CLIConfig[R]] extends CLIConfig[R] {

  private var _cudaArchitectures : Seq[Int] = Seq.empty
  def setCUDAArchitectures(archs : Seq[Int]) : R = {
    _cudaArchitectures = archs
    self
  }
  def cudaArchitectures : Seq[Int] = _cudaArchitectures

}
