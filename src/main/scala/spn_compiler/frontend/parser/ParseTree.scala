package spn_compiler.frontend.parser

/**
  * Parse tree, comprising the actual tree and, for convenience, a list of the input variables.
  * @param rootNode Root node of the parse tree structure.
  * @param inputVariables List of input variables.
  */
final case class ParseTree(rootNode : ParseTreeNode, inputVariables : List[InputVariableParseTree],
                           marginals : List[Set[NodeReferenceParseTree]]) {
  def validate() : Unit = {
    rootNode.validate()
    marginals.foreach(_.foreach(_.validate))
  }
}

/**
  * Super class of all parse tree nodes.
  */
sealed abstract class ParseTreeNode {
  def validate() : Unit
}

/**
  * Representation of a reference to some other [[ParseTreeNode]] not yet resolved.
  */
case object UnresolvedReferenceParseTree extends ParseTreeNode {
  override def validate(): Unit = throw new RuntimeException("Detected unresolved reference!")
}

/**
  * Reference to some other [[ParseTreeNode]].
  * @param id Unique identifier of the referenced node.
  */
case class NodeReferenceParseTree(id : String) extends ParseTreeNode {
  var declaration : ParseTreeNode = UnresolvedReferenceParseTree

  override def validate(): Unit = declaration.validate()
}

/**
  * Input variable.
  * @param name Uniquely identifying name of the variable.
  * @param index Index of the variable in the input vector.
  */
case class InputVariableParseTree(name : String, index : Int) extends ParseTreeNode {
  override def validate(): Unit = {} // Nothing to validate here.
}

/**
  * Sum node.
  * @param id Unique identifier.
  * @param addends Weighted (scaled) addends to this sum. Represented as list of tuples comprising
  *                the weights ([[Double]]) and a reference to another [[ParseTreeNode]].
  * @param childNodes [[ParseTreeNode]]s defined in the body of this node.
  */
case class SumNodeParseTree(id : String, addends : List[(Double, NodeReferenceParseTree)],
                            childNodes : List[ParseTreeNode]) extends ParseTreeNode {
  override def validate(): Unit = addends.foreach(a => a._2.validate())
}

/**
  * Product node.
  * @param id Unique identifier.
  * @param multiplicands List of references to other [[ParseTreeNode]]s used as multiplicands.
  * @param childNodes [[ParseTreeNode]]s defined in the body of this node.
  */
case class ProductNodeParseTree(id : String, multiplicands : List[NodeReferenceParseTree],
                                childNodes : List[ParseTreeNode]) extends ParseTreeNode {
  override def validate(): Unit = multiplicands.foreach(m => m.validate())
}

/**
  * Univariate Poisson distribution.
  * @param id Unique identifier.
  * @param inputVar Reference to the input variable.
  * @param lambda Value of the lambda parameter of this Poisson distribution.
  */
case class PoissonNodeParseTree(id : String, inputVar : NodeReferenceParseTree, lambda : Double) extends ParseTreeNode {
  override def validate(): Unit = inputVar.validate()
}

/**
  * Univariate distribution represented by a histogram.
  * @param id Unique identifier.
  * @param indexVar Reference to the indexing variable.
  * @param breaks List of lower bounds for each bucket plus the upper bound of the last bucket.
  * @param values List of values associated with each bucket.
  */
case class HistogramNodeParseTree(id : String, indexVar : NodeReferenceParseTree, breaks : List[Int],
                                  values : List[Double]) extends ParseTreeNode {
  override def validate(): Unit = indexVar.validate()
}
