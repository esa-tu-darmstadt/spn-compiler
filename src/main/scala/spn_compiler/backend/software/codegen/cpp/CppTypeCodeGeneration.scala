package spn_compiler.backend.software.codegen.cpp

import spn_compiler.backend.software.ast.nodes.types._
import spn_compiler.backend.software.codegen.TypeCodeGeneration

trait CppTypeCodeGeneration extends TypeCodeGeneration {

  def declareStructType(structType : StructType) : Unit = {
    val elements = structType.elements.
      map(e => "%s %s".format(generateType(e._2), e._1)).mkString("", ";", ";")
    writer.writeln("struct %s_t{%s};".format(structType.name, elements))
    writer.writeln("typedef struct %s_t %s;".format(structType.name, structType.name))
  }

  def generateType(ty : ASTType) : String = ty match {
    case IntegerType => "int"
    case RealType => "double"
    case VoidType => "void"
    case BooleanType => "bool"
    case StructType(name, _*) => name
    case ArrayType(elemType) => "%s*".format(generateType(elemType))
    case _ => throw new RuntimeException("No corresponding type specifier for type %s!".format(ty))
  }

  override def declareVariable(ty: ASTType, varName: String): String = ty match{
    case IntegerType => "int %s".format(varName)
    case RealType => "double %s".format(varName)
    case BooleanType => "bool %s".format(varName)
    case StructType(tyName, _) => "%s_t %s".format(tyName, varName)
    case ArrayType(elemType) => "%s %s[]".format(generateType(elemType), varName)
    case _ => throw new RuntimeException("Cannot declare variable fo type %s!".format(ty))
  }
}
