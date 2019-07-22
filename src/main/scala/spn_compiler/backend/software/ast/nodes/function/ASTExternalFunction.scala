package spn_compiler.backend.software.ast.nodes.function

import spn_compiler.backend.software.ast.nodes.types.{ASTType, StructType}

class ASTExternalHeader(val name : String, val local : Boolean = false)

object ASTExternalHeader {
  def unapply(arg: ASTExternalHeader): Option[(String, Boolean)] = Some(arg.name, arg.local)
}

class ASTExternalFunction private[ast](val header : ASTExternalHeader,
                                       name : String, returnType : ASTType,
                                       paramTypes : ASTType*) extends ASTFunctionPrototype(name, returnType, paramTypes:_*)

object ASTExternalFunction {

  def unapplySeq(func : ASTExternalFunction) : Option[(ASTExternalHeader, String, ASTType, Seq[ASTType])] =
    Some(func.header, func.name, func.returnType, func.getParameterTypes)

}

class ASTExternalStructType private[ast](val header : ASTExternalHeader, name : String,
                                         elements : List[(String, ASTType)]) extends StructType(name, elements)

object ASTExternalStructType {
  def unapplySeq(arg: ASTExternalStructType):   Option[(ASTExternalHeader, String, Seq[(String, ASTType)])] =
    Some(arg.header, arg.name, arg.elements)
}