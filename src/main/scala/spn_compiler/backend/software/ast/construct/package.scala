package spn_compiler.backend.software.ast

import spn_compiler.backend.software.ast.nodes.reference.{ASTReferencable, ASTReference, ASTVariableReference}
import spn_compiler.backend.software.ast.nodes.value.access.ASTVariableRead

package object construct {

  implicit def variable2Reference(variable : ASTReferencable) : ASTVariableReference = new ASTVariableReference(variable)

  implicit def reference2Read[Entity](entity : Entity)(implicit conv: Entity => ASTReference) : ASTVariableRead =
    new ASTVariableRead(conv(entity))

}
