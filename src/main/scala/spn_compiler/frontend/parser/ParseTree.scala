package spn_compiler.frontend.parser

final case class ParseTree(rootNode : ParseTreeNode, inputVariables : List[InputVariableParseTree])

/**
  * Super class of all parse tree nodes.
  */
sealed abstract class ParseTreeNode

/**
  * Representation of a reference to some other [[ParseTreeNode]] not yet resolved.
  */
case object UnresolvedReferenceParseTree extends ParseTreeNode

/**
  * Reference to some other [[ParseTreeNode]].
  * @param id Unique identifier of the referenced node.
  */
case class NodeReferenceParseTree(id : String) extends ParseTreeNode {
  var declaration : ParseTreeNode = UnresolvedReferenceParseTree
}

/**
  * Input variable.
  * @param name Uniquely identifying name of the variable.
  * @param index Index of the variable in the input vector.
  */
case class InputVariableParseTree(name : String, index : Int) extends ParseTreeNode

/**
  * Sum node.
  * @param id Unique identifier.
  * @param addends Weighted (scaled) addends to this sum. Represented as list of tuples comprising
  *                the weights ([[Double]]) and a reference to another [[ParseTreeNode]].
  * @param childNodes [[ParseTreeNode]]s defined in the body of this node.
  */
case class SumNodeParseTree(id : String, addends : List[(Double, NodeReferenceParseTree)],
                            childNodes : List[ParseTreeNode]) extends ParseTreeNode

/**
  * Product node.
  * @param id Unique identifier.
  * @param multiplicands List of references to other [[ParseTreeNode]]s used as multiplicands.
  * @param childNodes [[ParseTreeNode]]s defined in the body of this node.
  */
case class ProductNodeParseTree(id : String, multiplicands : List[NodeReferenceParseTree],
                                childNodes : List[ParseTreeNode]) extends ParseTreeNode

/**
  * Univariate Poisson distribution.
  * @param id Unique identifier.
  * @param inputVar Reference to the input variable.
  * @param lambda Value of the lambda parameter of this Poisson distribution.
  */
case class PoissonNodeParseTree(id : String, inputVar : NodeReferenceParseTree, lambda : Double) extends ParseTreeNode

/**
  * Univariate distribution represented by a histogram.
  * @param id Unique identifier.
  * @param indexVar Reference to the indexing variable.
  * @param breaks List of lower bounds for each bucket plus the upper bound of the last bucket.
  * @param values List of values associated with each bucket.
  */
case class HistogramNodeParseTree(id : String, indexVar : NodeReferenceParseTree, breaks : List[Int],
                                  values : List[Double]) extends ParseTreeNode
