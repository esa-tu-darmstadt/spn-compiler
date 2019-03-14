package spn_compiler.backend.software.ast.nodes.statement.control_flow

import spn_compiler.backend.software.ast.nodes.function.ASTFunctionPrototype
import spn_compiler.backend.software.ast.nodes.statement.ASTStatement
import spn_compiler.backend.software.ast.nodes.types.VoidType
import spn_compiler.backend.software.ast.nodes.value.ASTValue
import spn_compiler.backend.software.ast.nodes.value.function.ASTCallExpression

class ASTCallStatement private[ast](val call : ASTCallExpression) extends ASTStatement {
  if(call.function.returnType != VoidType){
    println("WARNING: Not using return value of function in call statement!") // TODO Logging.
  }
}

object ASTCallStatement {

  def unapplySeq(arg : ASTCallStatement) : Option[(ASTFunctionPrototype, Seq[ASTValue])] =
    ASTCallExpression.unapplySeq(arg.call)

}

class ASTReturnStatement private[ast](val returnValue : ASTValue) extends ASTStatement

object ASTReturnStatement {
  def unapply(arg: ASTReturnStatement): Option[ASTValue] = Some(arg.returnValue)
}
