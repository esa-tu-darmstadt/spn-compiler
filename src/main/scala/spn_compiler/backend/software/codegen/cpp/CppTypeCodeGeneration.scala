package spn_compiler.backend.software.codegen.cpp

import spn_compiler.backend.software.ast.nodes.types._
import spn_compiler.backend.software.codegen.TypeCodeGeneration

trait CppTypeCodeGeneration extends TypeCodeGeneration {

  def generateType(ty : ASTType) : String = ty match {
    case IntegerType => "int"
    case RealType => "double"
    case VoidType => "void"
    case BooleanType => "bool"
    case StructType(name, _) => "%s_t".format(name)
    case ArrayType(elemType) => "%s[]".format(generateType(elemType))
  }

}
