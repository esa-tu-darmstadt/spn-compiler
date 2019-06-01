package spn_compiler.server

import scopt._
import spn_compiler.driver.compile.cpu.CPUCompilerDriver
import spn_compiler.driver.config.{BaseConfig, CLIConfig, CPPCompileConfig}
import spn_compiler.driver.option.{BaseOptions, CPPCompileOptions}
import spn_compiler.frontend.parser.Parser
import spn_compiler.util.logging.Logging
import spn_compiler.util.statistics.GraphStatistics

import scala.concurrent.{ExecutionContext, Future}

import io.grpc.{Server, ServerBuilder}
import spn_compiler.server.grpc.spncserver.{CompileReply, CompileRequest, SPNCompilerGrpc}

class CompileServerConfig extends CLIConfig[CompileServerConfig] with BaseConfig[CompileServerConfig] with CPPCompileConfig[CompileServerConfig] {
  override def self: CompileServerConfig = this
}

object CompileServer extends App with Logging {

  private lazy val port = 50051

  override def main(args: Array[String]): Unit = {
    val server = new CompileServer(ExecutionContext.global)
    server.start()
    server.blockUntilShutdown()
  }

}

class CompileServer(executionContext: ExecutionContext) { self =>
  private[this] var server: Server = _

  private def start(): Unit = {
    server = ServerBuilder.forPort(CompileServer.port).addService(SPNCompilerGrpc.bindService(new SPNCompilerImpl, executionContext)).build.start
    CompileServer.info("Server started, listening on " + CompileServer.port)

    sys.addShutdownHook {
      CompileServer.info("Shutting down gRPC server since JVM is shutting down")
      self.stop()
    }
  }

  private def stop(): Unit = {
    if (server != null) {
      server.shutdown()
    }
  }

  private def blockUntilShutdown(): Unit = {
    if (server != null) {
      server.awaitTermination()
    }
  }

  private class SPNCompilerImpl extends SPNCompilerGrpc.SPNCompiler {
    override def sPNCompileText(req: CompileRequest) = {

      val spn = Parser.parseString(req.spn)

      val cliConfig = new CompileServerConfig().setVerbosityLevel(Logging.VerbosityVerbose)

      CPUCompilerDriver.execute(spn, cliConfig)

      if(cliConfig.computeStats){
        GraphStatistics.computeStatistics(spn, cliConfig.statsFile)
      }

      val reply = CompileReply(message = "Compiled.\n" + req.spn)
      Future.successful(reply)
    }

    override def sPNCompileJSON(req: CompileRequest) = {
      CompileServer.warn("WARNING: sPNCompileJSON -- Not implemented!")
      val reply = CompileReply(message = "Could NOT compile.\n" + req.spn)
      Future.successful(reply)
    }
  }

}