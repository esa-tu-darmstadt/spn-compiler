package spn_compiler.driver.compile.cuda

import java.io.File

import spn_compiler.driver.config.CompilerConfig
import spn_compiler.util.logging.Logging

import scala.collection.mutable.ListBuffer
import scala.sys.process.Process

case object NVCCCompilerDriver extends Logging {

  def compile[C <: CompilerConfig[C]](config: C, files: File*): Unit = {
    val flags : ListBuffer[String] = ListBuffer()
    flags.append(s"-O${config.optimizationLevel}")
    if(config.isFastMathEnabled){
      flags.append("--use_fast_math")
    }
    flags.append(s"-o ${config.outputFile.getAbsoluteFile.toString}")
    val cmd : String = "nvcc %s %s".format(flags.mkString(" "), files.map(_.getAbsoluteFile.toString).mkString(" "))
    val process = Process(cmd)
    try{
      debug(s"Running CUDA compiler nvcc with command: $cmd")
      val output = process.lineStream
      output.foreach(debug)
    }
    catch{
      case error : Exception => {
        this.error(s"Compilation error, running command $cmd")
        System.exit(-1)
      }
    }
    info(s"Successfully compiled CUDA software executable ${config.outputFile.getAbsoluteFile.toString}")
  }
}
