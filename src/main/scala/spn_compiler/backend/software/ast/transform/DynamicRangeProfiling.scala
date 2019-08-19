package spn_compiler.backend.software.ast.transform

import spn_compiler.backend.software.ast.nodes.function.ASTFunction
import spn_compiler.backend.software.ast.nodes.module.ASTModule
import spn_compiler.backend.software.ast.nodes.statement.ASTStatement
import spn_compiler.backend.software.ast.nodes.statement.control_flow.ASTReturnStatement
import spn_compiler.backend.software.ast.nodes.statement.variable.{ASTVariableAssignment, ASTVariableDeclaration}
import spn_compiler.backend.software.ast.nodes.types.{IntegerType, ScalarType}
import spn_compiler.backend.software.ast.nodes.value.ASTValue
import spn_compiler.backend.software.ast.nodes.value.access.ASTVariableRead
import spn_compiler.backend.software.ast.nodes.value.constant.ASTConstant
import spn_compiler.backend.software.ast.nodes.value.expression.{ASTAddition, ASTMultiplication}
import spn_compiler.backend.software.ast.predef.RegisterRange

object DynamicRangeProfiling {

  def transformAST(ast : ASTModule) : Unit = {
    val ASTModule(_, _, _, _, funcs) = ast
    funcs.filter(_.name == "spn").foreach(transformSPNFunction(_, ast))
  }

  private def transformSPNFunction(func : ASTFunction, module : ASTModule) : Unit = func.body.getAllStatements.foreach{
    case declare @ ASTVariableDeclaration(variable, Some(v)) =>
      func.body.replace(declare, module.declareVariable(variable, transformValue(v, module)))
    case assign @ ASTVariableAssignment(lhs, v) =>
      func.body.replace(assign, module.assignVariable(lhs, transformValue(v, module)))
    case ret @ ASTReturnStatement(v) =>
      func.body.replace(ret, module.ret(transformValue(v, module)))
    case s : ASTStatement => println(s)
  }

  private def transformValue(value : ASTValue, module : ASTModule) : ASTValue = value match {
    case ASTAddition(l, r) =>
      module.add(transformValue(l, module), transformValue(r, module))

    case ASTMultiplication(l, r) =>
      module.mul(transformValue(l, module), transformValue(r, module))

    case c : ASTConstant[ScalarType, AnyVal] => module.call(RegisterRange, module.constantValue(IntegerType, 0), c)

    case read : ASTVariableRead => module.call(RegisterRange,
      module.constantValue(IntegerType, read.getAnnotation("spn.graph.depth").get.asInstanceOf[Int]), read)
  }

}
