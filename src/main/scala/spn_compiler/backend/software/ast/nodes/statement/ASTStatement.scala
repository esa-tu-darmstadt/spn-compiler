package spn_compiler.backend.software.ast.nodes.statement

import spn_compiler.backend.software.ast.nodes.ASTNode

abstract class ASTStatement extends ASTNode {

  private var _block : Option[ASTBlockStatement] = None

  def block : Option[ASTBlockStatement] = _block

  private[ast] def setBlock(block : ASTBlockStatement) : Unit = {
    require(_block.isEmpty, "A statement can only be contained in one block, i.e. used a single time!")
    _block = Some(block)
  }

}