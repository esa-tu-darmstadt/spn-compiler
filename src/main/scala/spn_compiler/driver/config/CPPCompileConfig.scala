package spn_compiler.driver.config

import java.io.File

import spn_compiler.driver.compile.cpu.{CPPCompilerDriver, ClangCPPDriver, GCCCPPDriver}

trait CPPCompileConfig[R <: CLIConfig[R]] extends CLIConfig[R] {

  private var optLevel : Int = 0
  def setOptimizationLevel(level : Int) : R = {
    optLevel = Math.max(optLevel, level)
    self
  }
  def optimizationLevel : Int = optLevel

  private var outFile : File = new File("spn.out")
  def setOutFile(file : File) : R = {
    outFile = file
    self
  }
  def outputFile : File = outFile

  private var codeOnly : Boolean = false
  def setCodeOnly(bool : Boolean) : R = {
    codeOnly = bool
    self
  }
  def outputCodeOnly : Boolean = codeOnly

  private var fastMath : Boolean = false
  def enableFastMath(bool : Boolean) : R = {
    fastMath = bool
    self
  }
  def isFastMathEnabled : Boolean = fastMath

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

}

private[driver] object CPPCompileConfig{
  def availableCompilers : List[(String, CPPCompilerDriver)] =
    List(ClangCPPDriver.compilerName, GCCCPPDriver.compilerName)
}