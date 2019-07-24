package spn_compiler.driver.config

import java.io.File

import scala.collection.mutable.ListBuffer

trait CompilerConfig[R <: CLIConfig[R]] extends CLIConfig[R] {

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

  private val _macros : ListBuffer[String] = ListBuffer()
  def addMacro(_macro : String) : R = {
    _macros.append(_macro)
    self
  }
  def macros : List[String] = _macros.toList

  private val flags : ListBuffer[String] = ListBuffer()
  def addCompilerFlag(flag : String) : R = {
    flags.append(flag)
    self
  }
  def compilerFlags : List[String] = flags.toList

}
