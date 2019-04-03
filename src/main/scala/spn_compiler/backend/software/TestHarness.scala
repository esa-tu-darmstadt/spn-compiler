package spn_compiler.backend.software

import spn_compiler.backend.software.ast.construct.{ASTBuilder, _}
import spn_compiler.backend.software.ast.nodes.types.IntegerType

object TestHarness {

  val builder = new ASTBuilder
  val variable = builder.createVariable(IntegerType, "var")
  val ref = builder.referenceVariable(variable)
  val five = builder.constantValue(IntegerType, 5)
  val assign = builder.assignVariable(variable, five)

}
