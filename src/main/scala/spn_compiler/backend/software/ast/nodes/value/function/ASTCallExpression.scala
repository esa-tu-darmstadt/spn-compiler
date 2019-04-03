package spn_compiler.backend.software.ast.nodes.value.function

import spn_compiler.backend.software.ast.nodes.function.ASTFunctionPrototype
import spn_compiler.backend.software.ast.nodes.types.ASTType
import spn_compiler.backend.software.ast.nodes.value.ASTValue

class ASTCallExpression private[ast](val function : ASTFunctionPrototype, val parameters : ASTValue*) extends ASTValue {

  require((function.getParameterTypes zip parameters.map(_.getType)).forall{case (pt, at) => pt == at},
    "Actual parameter types must match formal parameter types!")

  override def getType: ASTType = function.returnType

}

object ASTCallExpression {

  def unapplySeq(call : ASTCallExpression) : Option[(ASTFunctionPrototype, Seq[ASTValue])] = Some(call.function, call.parameters)

}
