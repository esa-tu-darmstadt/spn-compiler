package spn_compiler.backend.software.ast.nodes

import scala.collection.mutable

abstract class ASTNode {

  private val annotations : mutable.Map[String, Any] = mutable.Map()

  def addAnnotation(key : String, value : Any) : this.type = {
    annotations += key -> value
    this
  }

  def getAnnotation(key : String) : Option[Any] = annotations.get(key)
  
}
