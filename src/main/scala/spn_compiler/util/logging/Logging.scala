package spn_compiler.util.logging

import wvlet.log.LogLevel.{DEBUG, INFO, TRACE}
import wvlet.log.{LogFormatter, Logger}

object Logging {

  private val logger = Logger("spnc")
  logger.setFormatter(LogFormatter.SourceCodeLogFormatter)

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

  def setVerbosityLevel(level : VerbosityLevel) : Unit = level match {
    case VerbosityNormal => logger.setLogLevel(INFO)
    case VerbosityVerbose => logger.setLogLevel(DEBUG)
    case VerbosityExtraVerbose => logger.setLogLevel(TRACE)
  }

}

trait Logging {
  def error(message: Any): Unit = Logging.logger.error(message)

  def error(message: Any, cause: Throwable): Unit = Logging.logger.error(message, cause)

  def warn(message: Any): Unit = Logging.logger.warn(message)

  def warn(message: Any, cause: Throwable): Unit = Logging.logger.warn(message, cause)

  def info(message: Any): Unit = Logging.logger.info(message)

  def info(message: Any, cause: Throwable): Unit = Logging.logger.info(message, cause)

  def debug(message: Any): Unit = Logging.logger.debug(message)

  def debug(message: Any, cause: Throwable): Unit = Logging.logger.debug(message, cause)

  def trace(message: Any): Unit = Logging.logger.trace(message)

  def trace(message: Any, cause: Throwable): Unit = Logging.logger.trace(message, cause)
}


