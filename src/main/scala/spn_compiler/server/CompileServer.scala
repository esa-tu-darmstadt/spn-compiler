package spn_compiler.server

import scopt._
import spn_compiler.driver.compile.cpu.CPUCompilerDriver
import spn_compiler.driver.config.{BaseConfig, CLIConfig, CPPCompileConfig}
import spn_compiler.driver.option.{BaseOptions, CPPCompileOptions}
import spn_compiler.frontend.parser.Parser
import spn_compiler.util.logging.Logging
import spn_compiler.util.statistics.GraphStatistics

import scala.concurrent.{ExecutionContext, Future}
import java.util.logging.Logger

import io.grpc.{Server, ServerBuilder}
import spn_compiler.server.grpc.spncserver.{CompileReply, CompileRequest, SPNCompilerGrpc}

class CompileServerConfig extends CLIConfig[CompileServerConfig] with BaseConfig[CompileServerConfig] with CPPCompileConfig[CompileServerConfig] {
  override def self: CompileServerConfig = this
}

object CompileServer extends App with Logging {

  System.err.println("CompileServer object")

  private lazy val logger = Logger.getLogger(classOf[CompileServer].getName)
  private lazy val port = 50051

  private lazy val builder = OParser.builder[CompileServerConfig]

  private lazy val cliParser : OParser[_, CompileServerConfig] = {
    import builder._
    OParser.sequence(
      programName("spnc"),
      head("spnc", "0.0.2"),
      BaseOptions.apply,
      CPPCompileOptions.apply
    )
  }

  private lazy val cliConfig: CompileServerConfig = OParser.parse(cliParser, args, new CompileServerConfig())
    .getOrElse(throw new RuntimeException("CLI Error!"))

  Logging.setVerbosityLevel(cliConfig.verbosityLevel)

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
    CompileServer.logger.info("Server started, listening on " + CompileServer.port)

    sys.addShutdownHook {
      System.err.println("*** shutting down gRPC server since JVM is shutting down")
      self.stop()
      System.err.println("*** server shut down")
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

      CPUCompilerDriver.execute(spn, CompileServer.cliConfig)

      if(CompileServer.cliConfig.computeStats){
        GraphStatistics.computeStatistics(spn, CompileServer.cliConfig.statsFile)
      }

      val reply = CompileReply(message = "Compiled.\n" + req.spn)
      Future.successful(reply)
    }

    override def sPNCompileJSON(req: CompileRequest) = {
      System.err.println("WARNING: sPNCompileJSON -- Not implemented!")
      val reply = CompileReply(message = "Compiled.\n" + req.spn)
      Future.successful(reply)
      System.err.println("WARNING: sPNCompileJSON -- Not implemented!")
    }
  }

}