package spn_compiler.driver.config

import java.io.File

trait CLIConfig[R]{
  def self : R
}

trait BaseConfig[R <: CLIConfig[R]] extends CLIConfig[R] {

  private var inputFile : File = new File("structure.spn")
  def setInputFile(file : File) : R = {
    inputFile = file
    self
  }

  def in : File = inputFile

  sealed trait VerbosityLevel{
    val level : Int
  }
  case object VerbosityNormal extends VerbosityLevel{
    override val level: Int = 0
  }
  case object VerbosityVerbose extends VerbosityLevel{
    override val level: Int = 1
  }
  case object VerbosityExtraVerbose extends VerbosityLevel{
    override val level: Int = 2
  }

  private var verbosity : VerbosityLevel = VerbosityNormal
  def setVerbosityLevel(level :VerbosityLevel) : R = {
    verbosity = if(level.level > verbosity.level) level else verbosity
    self
  }

  private var _computeStats : Boolean = false
  def enableStats(enable : Boolean) : R = {
    _computeStats = enable
    self
  }

  def computeStats : Boolean = _computeStats

  private var _statsFile : File = new File("stats.spns")
  def setStatsFile(file : File) : R = {
    _statsFile = file
    self
  }

  def statsFile : File = _statsFile
}
