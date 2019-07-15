package spn_compiler.backend.software.ast.predef

import spn_compiler.backend.software.ast.nodes.function.ASTExternalFunction
import spn_compiler.backend.software.ast.nodes.types.RealType

case object RegisterRange extends ASTExternalFunction("spn-compiler-rt.hpp", "register_range",
  RealType, RealType)
