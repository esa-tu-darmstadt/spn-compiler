package spn_compiler.backend.software.ast.nodes.statement

import spn_compiler.backend.software.ast.nodes.ASTNode

abstract class ASTStatement extends ASTNode {

  private var _block : Option[ASTBlockStatement] = None

  def block : ASTBlockStatement = _block.orNull // TODO: Return option, if interface usage allows/requires.

  private[ast] def setBlock(block : ASTBlockStatement) : Unit = {
    require(_block.isEmpty, "A statement can only be contained in one block, i.e. used a single time!")
    _block = Some(block)
  }

  def insertBefore(stmt : ASTStatement) : ASTStatement = {
    require(_block.isDefined, "Statement must be inserted into a block before you can insert statements before it!")
    _block.get.insertBefore(this, stmt)
  }

  def insertAfter(stmt : ASTStatement) : ASTStatement = {
    require(_block.isDefined, "Statement must be inserted into a block before you can insert statements after it!")
    _block.get.insertAfter(this, stmt)
  }

  def delete() : Unit = {
    if(_block.isDefined){
      _block.get.delete(this)
    }
  }

}