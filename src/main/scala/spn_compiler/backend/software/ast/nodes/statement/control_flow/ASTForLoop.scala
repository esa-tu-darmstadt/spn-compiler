package spn_compiler.backend.software.ast.nodes.statement.control_flow

import spn_compiler.backend.software.ast.extensions.openmp.constructs.OMPCanonicalLoop
import spn_compiler.backend.software.ast.nodes.reference.ASTReference
import spn_compiler.backend.software.ast.nodes.statement.{ASTBlockStatement, ASTStatement}
import spn_compiler.backend.software.ast.nodes.types.BooleanType
import spn_compiler.backend.software.ast.nodes.value.ASTValue

class ASTForLoop private[ast](val initVar : Option[ASTReference], val initValue : Option[ASTValue], val testValue : ASTValue,
                              val incrVar : Option[ASTReference], val incrValue : Option[ASTValue])
  extends ASTStatement with OMPCanonicalLoop {

  require(!(initVar.isDefined ^ initValue.isDefined), "Init variable and value must both be given or empty!")
  require(initVar.isEmpty || initVar.get.getType == initValue.get.getType, "Init variable and value must be of the same type!")
  require(!(incrVar.isDefined ^ incrValue.isDefined), "Increment variable and value must both be given or empty!")
  require(incrVar.isEmpty || incrVar.get.getType == incrValue.get.getType, "Increment variable and value must be of the same type!")
  require(testValue.getType==BooleanType, "Test expression must be of boolean type!")

  val body : ASTBlockStatement = new ASTBlockStatement

}

object ASTForLoop {

  def unapply(arg: ASTForLoop): Option[(Option[ASTReference], Option[ASTValue],
    ASTValue, Option[ASTReference], Option[ASTValue], ASTBlockStatement)] =
    Some(arg.initVar, arg.initValue, arg.testValue, arg.incrVar, arg.incrValue, arg.body)

}
