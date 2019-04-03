package spn_compiler.backend.software.codegen

import spn_compiler.backend.software.ast.nodes.reference.ASTReference
import spn_compiler.backend.software.ast.nodes.types.ASTType
import spn_compiler.backend.software.ast.nodes.value.ASTValue

trait ReferenceCodeGeneration {

  def generateReference(reference : ASTReference) : String

}

trait TypeCodeGeneration {

  def generateType(ty : ASTType) : String

  def declareVariable(ty : ASTType, varName : String) : String

}

trait ValueCodeGeneration {

  def generateValue(value : ASTValue) : String

}