package spn_compiler.backend.software.codegen

import spn_compiler.backend.software.ast.nodes.reference.ASTReference
import spn_compiler.backend.software.ast.nodes.types.{ASTType, StructType}
import spn_compiler.backend.software.ast.nodes.value.ASTValue

trait CodeGenerator {

  def writer : CodeWriter

}

trait ReferenceCodeGeneration {

  def generateReference(reference : ASTReference) : String

}

trait TypeCodeGeneration extends CodeGenerator {

  def generateType(ty : ASTType) : String

  def declareVariable(ty : ASTType, varName : String) : String

  def declareStructType(structType : StructType) : Unit

}

trait ValueCodeGeneration {

  def generateValue(value : ASTValue) : String

}