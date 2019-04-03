package spn_compiler.backend.software.ast.nodes.statement.control_flow

import spn_compiler.backend.software.ast.nodes.statement.{ASTBlockStatement, ASTStatement}
import spn_compiler.backend.software.ast.nodes.types.BooleanType
import spn_compiler.backend.software.ast.nodes.value.ASTValue

class ASTIfStatement private[ast](val testExpression : ASTValue) extends ASTStatement {
  require(testExpression.getType==BooleanType, "Test expression must be of boolean type!")

  val thenBranch : ASTBlockStatement = new ASTBlockStatement

  val elseBranch : ASTBlockStatement = new ASTBlockStatement

}


object ASTIfStatement {

  def unapply(arg: ASTIfStatement): Option[(ASTValue, ASTBlockStatement, ASTBlockStatement)] =
    Some(arg.testExpression, arg.thenBranch, arg.elseBranch)

}