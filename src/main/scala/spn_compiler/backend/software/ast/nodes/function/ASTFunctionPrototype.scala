package spn_compiler.backend.software.ast.nodes.function

import spn_compiler.backend.software.ast.nodes.ASTNode
import spn_compiler.backend.software.ast.nodes.types.ASTType

class ASTFunctionPrototype private[ast](val name : String, val returnType : ASTType, private val paramTypes : ASTType*)
  extends ASTNode {

  def getParameterTypes : List[ASTType] = paramTypes.toList

}

object ASTFunctionPrototype {

  def unapplySeq(func : ASTFunctionPrototype) : Option[(String, ASTType, Seq[ASTType])] =
    Some(func.name, func.returnType, func.paramTypes)

}