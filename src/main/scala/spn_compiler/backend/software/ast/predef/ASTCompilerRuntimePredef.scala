package spn_compiler.backend.software.ast.predef

import spn_compiler.backend.software.ast.nodes.function.{ASTExternalFunction, ASTExternalHeader}
import spn_compiler.backend.software.ast.nodes.types.RealType

case object CompilerRuntimeHeader extends ASTExternalHeader("spn-compiler-rt.hpp", true)

case object RegisterRange extends ASTExternalFunction(CompilerRuntimeHeader, "register_range",
  RealType, RealType)
