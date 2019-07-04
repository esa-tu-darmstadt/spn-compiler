package spn_compiler.graph_ir.nodes

/**
  * IR graph, comprising the actual SPN and, for convenience, a list of input variables.
  * @param rootNode Root node of the SPN.
  * @param inputVariables List of input variables.
  */
final case class IRGraph(rootNode : IRNode, inputVariables : List[InputVar])

/**
  * Super-class of all nodes in the graph-representation of SPNs.
  * @param _identifier Unique string identifier of the node.
  */
sealed abstract class IRNode(_identifier : String) {

  private val identifier : String = _identifier

  /**
    * Accessor.
    * @return The unique identifer of the node.
    */
  def getID : String = identifier

  // TODO: Annotation system.
}

/**
  * Input variable to the SPN.
  * @param id see [[IRNode._identifier]].
  * @param index Index of this variable in the input vector.
  */
case class InputVar(id : String, index : Int) extends IRNode(id)

/**
  * Univariate Poisson distribution.
  * @param id see [[IRNode._identifier]].
  * @param variable [[InputVar]] associated with this distribution.
  * @param lambda Value of lambda in the Poisson formula.
  */
case class PoissonDistribution(id : String, variable : InputVar, lambda : Double) extends IRNode(id)

/**
  * Bucket in histogram, comprising interval covered by the bucket and the associated value.
  * The interval is defined as [lb, ub)
  * @param lowerBound Inclusive lower bound.
  * @param upperBound Exclusive upper bound.
  * @param value Associated value.
  */
final case class HistogramBucket(lowerBound : Int, upperBound : Int, value : Double)

/**
  * Histogram representing a univariate distribution.
  * @param id see [[IRNode._identifier]].
  * @param indexVar [[InputVar]] associated with this distribution.
  * @param buckets [[HistogramBucket]]s in the histogram.
  */
case class Histogram(id : String, indexVar : InputVar, buckets : List[HistogramBucket]) extends IRNode(id)

/**
  * Represents a marginalized input variable.
  * @param id see [[IRNode._identifier]]
  */
case class Marginal(id : String) extends IRNode(id)

/**
  * Representation of an weighted (scaled) addend to a [[WeightedSum]].
  * @param addend [[IRNode]] used as addend.
  * @param weight Weight (scaling factor).
  */
final case class WeightedAddend(addend : IRNode, weight : Double = 1.0) extends IRNode(addend.getID+"_w")

/**
  * Weighted (scaled), n-ary sum.
  * @param id see [[IRNode._identifier]].
  * @param addends [[WeightedAddend]]s to this sum.
  */
case class WeightedSum(id : String, addends : List[WeightedAddend]) extends IRNode(id)

/**
  * Non-weighted, n-ary sum.
  * @param id see [[IRNode._identifier]]
  * @param addends [[IRNode]]s used as addends to this sum.
  */
case class Sum(id : String, addends : List[IRNode]) extends IRNode(id)

/**
  * N-ary product (multiplication).
  * @param id see [[IRNode._identifier]].
  * @param multiplicands [[IRNode]]s used as multiplicands to this sum.
  */
case class Product(id : String, multiplicands : List[IRNode]) extends IRNode(id)



