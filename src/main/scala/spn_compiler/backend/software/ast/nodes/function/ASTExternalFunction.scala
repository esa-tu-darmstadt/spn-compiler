package spn_compiler.backend.software.ast.nodes.function

import spn_compiler.backend.software.ast.nodes.types.ASTType

class ASTExternalFunction private[ast](val header : String, name : String, returnType : ASTType, paramTypes : ASTType*)
  extends ASTFunctionPrototype(name, returnType, paramTypes:_*)

object ASTExternalFunction {

  def unapplySeq(func : ASTExternalFunction) : Option[(String, String, ASTType, Seq[ASTType])] =
    Some(func.header, func.name, func.returnType, func.getParameterTypes)

}