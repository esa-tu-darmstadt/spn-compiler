package spn_compiler.backend.software.codegen.cpp

import spn_compiler.backend.software.ast.extensions.cpp.value.{CPPAddressOfOperator, CPPSizeOfOperator}
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
    case ASTCallExpression(func, params @ _*) =>
      "%s(%s)".format(func.name, params.map(generateValue).mkString(","))
    case const : ASTConstant[ScalarType, AnyVal] => const.getConstantValue.toString
    case ASTArrayInit(values @ _*) => "{%s}".format(values.map(generateValue).mkString(","))
    case div @ ASTDivision(l, r) => "%s / %s".format(formatPrecedence(l, div), formatPrecedence(r, div))
    case or @ ASTOr(l, r) => "%s | %s".format(formatPrecedence(l, or), formatPrecedence(r, or))
    case and @ ASTAnd(l, r) => "%s & %s".format(formatPrecedence(l, and), formatPrecedence(r, and))
    case xor @ ASTXor(l, r) => "%s ^ %s".format(formatPrecedence(l, xor), formatPrecedence(r, xor))
    case add @ ASTAddition(l, r) => "%s + %s".format(formatPrecedence(l, add), formatPrecedence(r, add))
    case rem @ ASTRemainder(l, r) => "%s %% %s".format(formatPrecedence(l, rem), formatPrecedence(r, rem))
    case sub @ ASTSubtraction(l, r) => "%s - %s".format(formatPrecedence(l, sub), formatPrecedence(r, sub))
    case mul @ ASTMultiplication(l, r) => "%s * %s".format(formatPrecedence(l, mul), formatPrecedence(r, mul))
    case comp @ ASTCmpGT(l, r) => "%s > %s".format(formatPrecedence(l, comp), formatPrecedence(r, comp))
    case comp @ ASTCmpGE(l, r) => "%s >= %s".format(formatPrecedence(l, comp), formatPrecedence(r, comp))
    case comp @ ASTCmpLT(l, r) => "%s < %s".format(formatPrecedence(l, comp), formatPrecedence(r, comp))
    case comp @ ASTCmpLE(l, r) => "%s <= %s".format(formatPrecedence(l, comp), formatPrecedence(r, comp))
    case comp @ ASTCmpEQ(l, r) => "%s == %s".format(formatPrecedence(l, comp), formatPrecedence(r, comp))
    case comp @ ASTCmpNE(l, r) => "%s != %s".format(formatPrecedence(l, comp), formatPrecedence(r, comp))
    case ASTTypeConversion(srcVal, tty) => "((%s) %s)".format(generateType(tty), generateValue(srcVal))
    case not @ ASTNot(op) => "~%s".format(formatPrecedence(op, not))
    case neg @ ASTNeg(op) => "-%s".format(formatPrecedence(op, neg))
    case ASTStructInit(_, values @ _*) => "{%s}".format(values.map(generateValue).mkString(","))
    case ASTVariableRead(_, reference) => generateReference(reference)
    case CPPAddressOfOperator(reference) => "&%s".format(generateReference(reference))
    case CPPSizeOfOperator(reference) => "sizeof(%s)".format(generateReference(reference))
  }

  private def formatPrecedence(operand : ASTValue, operator : ASTValue) : String =
    // HINT: If we want to preserve the associativity strictly as given by the
    // constructed AST, we would have to use greater-equal here.
    if(cppOperatorPrecedence(operand) > cppOperatorPrecedence(operator)){
      "(%s)".format(generateValue(operand))
    } else {
      generateValue(operand)
    }

  private def cppOperatorPrecedence(operator : ASTValue) : Int = operator match {
    // Source: https://de.cppreference.com/w/cpp/language/operator_precedence
    case c : ASTCallExpression => 0
    case const : ASTConstant[ScalarType, AnyVal] => 0
    case init : ASTArrayInit => 0
    case div : ASTDivision => 5
    case or : ASTOr => 12 // Bitwise
    case and : ASTAnd => 10 // Bitwise
    case xor : ASTXor => 11 // Bitwise
    case add : ASTAddition => 6
    case rem : ASTRemainder => 5
    case sub : ASTSubtraction => 6
    case mult : ASTMultiplication => 5
    case ASTCmpGT(l, r) => 8
    case ASTCmpGE(l, r) => 8
    case ASTCmpLT(l, r) => 8
    case ASTCmpLE(l, r) => 8
    case ASTCmpEQ(l, r) => 9
    case ASTCmpNE(l, r) => 9
    case ASTTypeConversion(srcVal, tty) => 0
    case ASTNot(op) => 3
    case ASTNeg(op) => 3
    case ASTStructInit(_, values @ _*) => 0
    case ASTVariableRead(_, reference) => 2 // Array access and member access
    case CPPAddressOfOperator(_) => 3
    case CPPSizeOfOperator(_) => 3
  }

}
