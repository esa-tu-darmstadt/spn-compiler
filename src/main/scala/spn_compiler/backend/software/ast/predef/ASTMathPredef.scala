package spn_compiler.backend.software.ast.predef

import spn_compiler.backend.software.ast.nodes.function.{ASTExternalFunction, ASTExternalHeader}
import spn_compiler.backend.software.ast.nodes.types.RealType

case object CMathHeader extends ASTExternalHeader("cmath")

case object Ceil extends ASTExternalFunction(CMathHeader, "ceil", RealType, RealType)
