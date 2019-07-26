package spn_compiler.backend.software.ast.predef

import spn_compiler.backend.software.ast.nodes.function.{ASTExternalFunction, ASTExternalHeader, ASTExternalStructType}
import spn_compiler.backend.software.ast.nodes.types._

case object CMathHeader extends ASTExternalHeader("cmath")

case object Ceil extends ASTExternalFunction(CMathHeader, "ceil", RealType, RealType)

case object LNSHeader extends ASTExternalHeader("spn-lns.hpp", true)

case object LNSType extends StructType("lns_t",
  List(("exp", PreciseIntegerType), ("sign", BooleanType), ("zero", BooleanType)))

case object LNSInit extends ASTExternalFunction(LNSHeader, "initializeInterpolator", VoidType,
  IntegerType, IntegerType, RealType)

case object LNS2Double extends ASTExternalFunction(LNSHeader, "lns_get_value", RealType, LNSType)

case object PositHeader extends ASTExternalHeader("spn-posit.hpp", true)

case object PositType extends ASTExternalStructType(PositHeader, "posit_t", List(("exp", RealType)))

case object FPSimHeader extends ASTExternalHeader("spn-fp.hpp", true)

case object FPSimType extends ASTExternalStructType(FPSimHeader, "spn_float_t", List(("d", RealType)))