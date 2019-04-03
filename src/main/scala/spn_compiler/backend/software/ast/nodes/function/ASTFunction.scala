package spn_compiler.backend.software.ast.nodes.function

import spn_compiler.backend.software.ast.nodes.reference.ASTReferencable
import spn_compiler.backend.software.ast.nodes.statement.ASTBlockStatement
import spn_compiler.backend.software.ast.nodes.types.ASTType

class ASTFunction private[ast](name : String, returnType : ASTType, private val params : ASTFunctionParameter*)
  extends ASTFunctionPrototype(name, returnType, params.map(_.ty):_*){

  def getParameters : List[ASTFunctionParameter] = params.toList

  val body = new ASTBlockStatement

}

object ASTFunction {

  def unapplySeq(func : ASTFunction) : Option[(String, ASTType, Seq[ASTFunctionParameter])] =
    Some(func.name, func.returnType, func.params)

}

class ASTFunctionParameter private[ast](val name : String, val ty : ASTType) extends ASTReferencable {

  override def getType: ASTType = ty

}

object ASTFunctionParameter {

  def unapply(arg: ASTFunctionParameter): Option[(String, ASTType)] = Some(arg.name, arg.ty)

}
