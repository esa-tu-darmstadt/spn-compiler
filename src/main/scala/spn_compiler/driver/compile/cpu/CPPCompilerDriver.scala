package spn_compiler.driver.compile.cpu

import java.io.File

import spn_compiler.driver.config.{CPPCompileConfig, CompilerConfig}
import spn_compiler.util.logging.Logging

import scala.collection.mutable.ListBuffer
import scala.sys.process.Process

trait CPPCompilerDriver {

  def compile[C <: CPPCompileConfig[C] with CompilerConfig[C]](config : C, files : File*)

  def compilerName : (String, CPPCompilerDriver)
}

case object ClangCPPDriver extends CPPCompilerDriver with Logging {

  override def compile[C <: CPPCompileConfig[C] with CompilerConfig[C]](config: C, files: File*): Unit = {
    val flags : ListBuffer[String] = ListBuffer()
    flags.append(s"-O${config.optimizationLevel}")
    if(config.isFastMathEnabled){
      flags.append("-ffast-math")
    }
    if(config.isOMPParallelForEnabled){
      flags.append("-fopenmp")
    }
    flags.append(config.macros.map(m => s"-D$m").mkString(" "))
    flags.append(config.compilerFlags.mkString(" "))
    if(config.isRangeProfilingEnabled){
      flags.append("-DSPN_PROFILE")
    }
    if(config.isLNSSimulationEnabled){
      flags.append(s"-DLNS_INTEGER_BITS=${config.lnsIntegerBits}")
      flags.append(s"-DLNS_FRACTION_BITS=${config.lnsFractionBits}")
    }
    if(config.isPositSimulationEnabled){
      flags.append(s"-DPOSIT_SIZE_N=${config.positSizeN}")
      flags.append(s"-DPOSIT_SIZE_ES=${config.positSizeES}")
    }
    if(config.isFPSimulationEnabled){
      flags.append(s"-DSPN_FLOAT_MANTISSA=${config.fpMantissa}")
      flags.append(s"-DSPN_E_MIN=${config.fpMaxExponent}")
      flags.append(s"-DSPN_E_MAX=${config.fpMinExponent}")
    }
    flags.append(s"-o ${config.outputFile.getAbsoluteFile.toString}")
    val cmd : String = "clang++ %s %s".format(flags.mkString(" "), files.map(_.getAbsoluteFile.toString).mkString(" "))
    val process = Process(cmd)
    try{
      debug(s"Running C++ compiler clang++ with command: $cmd")
      val output = process.lineStream
      output.foreach(debug)
    }
    catch{
      case error : Exception => {
        this.error(s"Compilation error, running command $cmd")
        System.exit(-1)
      }
    }
    info(s"Successfully compiled CPU software executable ${config.outputFile.getAbsoluteFile.toString}")
  }

  override def compilerName: (String, CPPCompilerDriver) = ("clang", ClangCPPDriver)
}

case object GCCCPPDriver extends CPPCompilerDriver with Logging {

  override def compile[C <: CPPCompileConfig[C] with CompilerConfig[C]](config: C, files: File*): Unit = {
    val flags : ListBuffer[String] = ListBuffer()
    flags.append(s"-O${config.optimizationLevel}")
    if(config.isFastMathEnabled){
      flags.append("-ffast-math")
    }
    if(config.isOMPParallelForEnabled){
      flags.append("-fopenmp")
    }
    flags.append(config.macros.map(m => s"-D$m").mkString(" "))
    flags.append(config.compilerFlags.mkString(" "))
    if(config.isRangeProfilingEnabled){
      flags.append("-DSPN_PROFILE")
    }
    if(config.isLNSSimulationEnabled){
      flags.append(s"-DLNS_INTEGER_BITS=${config.lnsIntegerBits}")
      flags.append(s"-DLNS_FRACTION_BITS=${config.lnsFractionBits}")
    }
    if(config.isPositSimulationEnabled){
      flags.append(s"-DPOSIT_SIZE_N=${config.positSizeN}")
      flags.append(s"-DPOSIT_SIZE_ES=${config.positSizeES}")
    }
    if(config.isFPSimulationEnabled){
      flags.append(s"-DSPN_FLOAT_MANTISSA=${config.fpMantissa}")
      flags.append(s"-DSPN_E_MIN=${config.fpMaxExponent}")
      flags.append(s"-DSPN_E_MAX=${config.fpMinExponent}")
    }
    flags.append(s"-o ${config.outputFile.getAbsoluteFile.toString}")
    val cmd : String = "g++ %s %s".format(flags.mkString(" "), files.map(_.getAbsoluteFile.toString).mkString(" "))
    val process = Process(cmd)
    try{
      debug(s"Running C++ compiler g++ with command: $cmd")
      val output = process.lineStream
      output.foreach(debug)
    }
    catch{
      case error : Exception => {
        this.error(s"Compilation error, running command $cmd")
        System.exit(-1)
      }
    }
    info(s"Successfully compiled CPU software executable ${config.outputFile.getAbsoluteFile.toString}")
  }

  override def compilerName: (String, CPPCompilerDriver) = ("gcc", GCCCPPDriver)
}