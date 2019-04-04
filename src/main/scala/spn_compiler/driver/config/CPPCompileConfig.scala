package spn_compiler.driver.config

trait CPPCompileConfig[R <: CLIConfig[R]] extends CLIConfig[R] {

  private var optimizationLevel : Int = 0
  def setOptimizationLevel(level : Int) : R = {
    optimizationLevel = Math.max(optimizationLevel, level)
    self
  }
}
