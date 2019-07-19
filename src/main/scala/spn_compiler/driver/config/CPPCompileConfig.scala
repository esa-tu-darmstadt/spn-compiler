package spn_compiler.driver.config

import spn_compiler.driver.compile.cpu.{CPPCompilerDriver, ClangCPPDriver, GCCCPPDriver}

trait CPPCompileConfig[R <: CLIConfig[R]] extends CLIConfig[R] {

  private var compiler : CPPCompilerDriver = ClangCPPDriver
  def setCompiler(cxx : CPPCompilerDriver) : R = {
    compiler = cxx
    self
  }
  def compilerDriver : CPPCompilerDriver = compiler

  private var ompParFor : Boolean = false
  def enableOMPParallelFor(bool : Boolean) : R = {
    ompParFor = bool
    self
  }
  def isOMPParallelForEnabled : Boolean = ompParFor

  private var rangeProfiling : Boolean = false
  def enableRangeProfiling(bool : Boolean) : R = {
    rangeProfiling = bool
    self
  }
  def isRangeProfilingEnabled : Boolean = rangeProfiling

}

private[driver] object CPPCompileConfig{
  def availableCompilers : List[(String, CPPCompilerDriver)] =
    List(ClangCPPDriver.compilerName, GCCCPPDriver.compilerName)
}