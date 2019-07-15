package spn_compiler.backend.software.ast.nodes.function

import spn_compiler.backend.software.ast.nodes.types.ASTType

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