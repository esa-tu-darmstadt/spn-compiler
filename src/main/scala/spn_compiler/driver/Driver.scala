package spn_compiler.driver

import java.nio.file.{Files, Paths}

import spn_compiler.frontend.parser.Parser

object Driver extends App {

  if(args.length != 1){
    throw new RuntimeException("Expecting a single file as input!")
  }

  if(!Files.exists(Paths.get(args(0)))){
    throw new RuntimeException("Specified input file does not exist!")
  }

  val spn = Parser.parseFile(args(0))
}
