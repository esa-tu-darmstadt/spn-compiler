package spn_compiler.backend.software.codegen.cpp

import spn_compiler.backend.software.ast.nodes.reference.{ASTElementReference, ASTIndexReference, ASTReference, ASTVariableReference}
import spn_compiler.backend.software.codegen.ValueCodeGeneration

trait CppReferenceCodeGeneration extends ValueCodeGeneration {

  def generateReference(reference : ASTReference) : String = reference match {
    case ASTVariableReference(_ , variable) => variable.name
    case ASTIndexReference(_, ref, index) => "%s[%s]".format(generateReference(ref), generateValue(index))
    case ASTElementReference(_, ref, elem) => "%s.%s".format(generateReference(ref), elem)
  }

}
