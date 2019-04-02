package spn_compiler.backend.software.codegen.cpp

import spn_compiler.backend.software.ast.nodes.types.ScalarType
import spn_compiler.backend.software.ast.nodes.value.ASTValue
import spn_compiler.backend.software.ast.nodes.value.access.ASTVariableRead
import spn_compiler.backend.software.ast.nodes.value.constant.{ASTArrayInit, ASTConstant, ASTStructInit}
import spn_compiler.backend.software.ast.nodes.value.expression._
import spn_compiler.backend.software.ast.nodes.value.function.ASTCallExpression
import spn_compiler.backend.software.ast.nodes.value.type_conversion.ASTTypeConversion
import spn_compiler.backend.software.codegen.{ReferenceCodeGeneration, TypeCodeGeneration}

trait CppValueCodeGeneration extends ReferenceCodeGeneration with TypeCodeGeneration {

  def generateValue(value : ASTValue) : String = value match {
    case ASTCallExpression(func, params : Seq[ASTValue]) =>
      "%s(%s)".format(func.name, params.map(generateValue).mkString(","))
    case const : ASTConstant[ScalarType, AnyVal] => const.getConstantValue.toString
    case ASTArrayInit(values : Seq[ASTValue]) => "{%s}".format(values.map(generateValue).mkString(","))
    case ASTDivision(l, r) => "(%s) / (%s)".format(generateValue(l), generateValue(r))
    case ASTOr(l, r) => "(%s) | (%s)".format(generateValue(l), generateValue(r))
    case ASTAnd(l, r) => "(%s) & (%s)".format(generateValue(l), generateValue(r))
    case ASTXor(l, r) => "(%s) ^ (%s)".format(generateValue(l), generateValue(r))
    case ASTAddition(l, r) => "(%s) + (%s)".format(generateValue(l), generateValue(r))
    case ASTRemainder(l, r) => "(%s) %% (%s)".format(generateValue(l), generateValue(r))
    case ASTSubtraction(l, r) => "(%s) - (%s)".format(generateValue(l), generateValue(r))
    case ASTMultiplication(l, r) => "(%s) * (%s)".format(generateValue(l), generateValue(r))
    case ASTCmpGT(l, r) => "(%s) > (%s)".format(generateValue(l), generateValue(r))
    case ASTCmpGE(l, r) => "(%s) >= (%s)".format(generateValue(l), generateValue(r))
    case ASTCmpLT(l, r) => "(%s) < (%s)".format(generateValue(l), generateValue(r))
    case ASTCmpLE(l, r) => "(%s) <= (%s)".format(generateValue(l), generateValue(r))
    case ASTCmpEQ(l, r) => "(%s) == (%s)".format(generateValue(l), generateValue(r))
    case ASTCmpNE(l, r) => "(%s) != (%s)".format(generateValue(l), generateValue(r))
    case ASTTypeConversion(srcVal, tty) => "((%s) %s)".format(generateType(tty), generateValue(srcVal))
    case ASTNot(op) => "~(%s)".format(generateValue(op))
    case ASTNeg(op) => "-(%s)".format(generateValue(op))
    case ASTStructInit(_, values : Seq[ASTValue]) => "{%s}".format(values.map(generateValue).mkString(","))
    case ASTVariableRead(_, reference) => generateReference(reference)
  }

}
