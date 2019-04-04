package spn_compiler.util.logging

import wvlet.log.LogLevel.DEBUG
import wvlet.log.{LogFormatter, Logger}

object Logging {
  private val logger = Logger("spnc")
  logger.setFormatter(LogFormatter.SourceCodeLogFormatter)
  logger.setLogLevel(DEBUG)
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


