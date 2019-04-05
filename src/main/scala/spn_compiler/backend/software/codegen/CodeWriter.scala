package spn_compiler.backend.software.codegen

import java.io.{BufferedWriter, File, FileWriter}

case class CodeWriter(file : File) {

  private val writer = new BufferedWriter(new FileWriter(file))

  def write(text : String) : Unit = writer.write(text)

  def writeln(text : String) : Unit = writer.write(text+"\n")

  def close() : Unit = writer.close()

}