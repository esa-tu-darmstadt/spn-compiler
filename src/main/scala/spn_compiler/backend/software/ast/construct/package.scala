package spn_compiler.backend.software.ast

import spn_compiler.backend.software.ast.nodes.reference.{ASTReference, ASTVariableReference}
import spn_compiler.backend.software.ast.nodes.value.access.ASTVariableRead
import spn_compiler.backend.software.ast.nodes.variable.ASTVariable

package object construct {

  implicit def variable2Reference(variable : ASTVariable) : ASTVariableReference = new ASTVariableReference(variable)

  implicit def reference2Read[Entity](entity : Entity)(implicit conv: Entity => ASTReference) : ASTVariableRead =
    new ASTVariableRead(conv(entity))

}
