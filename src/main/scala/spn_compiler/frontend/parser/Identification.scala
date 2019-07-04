package spn_compiler.frontend.parser

import scala.collection.mutable

/**
  * Establishes a link between a [[NodeReferenceParseTree]] and the corresponding
  * defining [[ParseTreeNode]].
  */
class Identification {

  private val backpatchEntries : mutable.Map[String, mutable.MutableList[NodeReferenceParseTree]] = mutable.Map()

  private val resolvedReferences : mutable.Map[String, ParseTreeNode] = mutable.Map()

  def performIdentification(parseTree: ParseTree) : Unit = {
    parseTree.inputVariables.foreach(v => resolvedReferences += v.name -> v)
    processGraph(parseTree.rootNode)
    parseTree.marginals.foreach(_.foreach(resolveReference))
  }

  private def processGraph(rootNode : ParseTreeNode) : Unit = rootNode match {
    case pn @ PoissonNodeParseTree(id, inputVarRef, _) => {
      resolveReference(inputVarRef)
      backpatchAndMarkResolved(id, pn)
    }

    case hn @ HistogramNodeParseTree(id, inputVarRef, _, _) => {
      resolveReference(inputVarRef)
      backpatchAndMarkResolved(id, hn)
    }

    case ws @ SumNodeParseTree(id, addends, children) => {
      children.foreach(c => processGraph(c))
      addends.foreach(a => resolveReference(a._2))
      backpatchAndMarkResolved(id, ws)
    }

    case p  @ ProductNodeParseTree(id, multiplicands, children) => {
      children.foreach(c => processGraph(c))
      multiplicands.foreach(m => resolveReference(m))
      backpatchAndMarkResolved(id, p)
    }

    case i : InputVariableParseTree =>
      // Nothing to do for InputVariables, they have been marked as resolved earlier.
  }

  private def backpatchAndMarkResolved(id : String, node : ParseTreeNode) : Unit = {
    if(backpatchEntries.contains(id)){
      backpatchEntries(id).foreach(_.declaration = node)
    }
    resolvedReferences += id -> node
  }

  private def resolveReference(reference : NodeReferenceParseTree) : Unit = {
    val id : String = reference.id
    if(resolvedReferences.contains(id)){
      reference.declaration = resolvedReferences(id)
    }
    else{
      backpatchEntries.getOrElseUpdate(id, mutable.MutableList()) += reference
    }
  }

}
