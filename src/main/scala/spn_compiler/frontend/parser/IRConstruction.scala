package spn_compiler.frontend.parser

import spn_compiler.graph_ir.nodes._

import scala.collection.mutable

/**
  * Constructs the actual graphical IR from a parse tree after [[Identification]].
  * @param parseTree [[ParseTree]] to construct IR from.
  */
class IRConstruction(private val parseTree : ParseTree) {

  private val resolvedNodes : mutable.Map[ParseTreeNode, IRNode] = mutable.Map()

  /**
    * Construct a [[IRGraph]] from the parse tree.
    * @return [[IRGraph]] comprising the SPN and the list of input variables.
    */
  def constructIRGraph : IRGraph = {
    val inputVariables = parseTree.inputVariables.map(constructSubTree(_).asInstanceOf[InputVar])
    val spnRoot = constructSubTree(parseTree.rootNode)
    IRGraph(spnRoot, inputVariables)
  }

  private def constructSubTree(subTreeRoot : ParseTreeNode) : IRNode = {
    resolvedNodes.getOrElseUpdate(subTreeRoot, subTreeRoot match{
      // Poisson Node
      case PoissonNodeParseTree(id, inputVar, lambda) => {
        val inputVariable = constructSubTree(inputVar)
        PoissonDistribution(id, inputVariable.asInstanceOf[InputVar], lambda)
      }
      // Histogram
      case HistogramNodeParseTree(id, inputVar, breaks, values) => {
        val indexVar = constructSubTree(inputVar)
        require(breaks.size==(values.size+1), "Cannot construct histogram buckets from given breaks and values!")
        // Zip all elements but the last (init) with all elements but the first.
        // Result is a list of tuples with lower and upper bounds for each bucket.
        val listBreaks = breaks.init zip breaks.tail
        val buckets = (listBreaks zip values).map{case ((lb, ub), v) => HistogramBucket(lb, ub, v)}
        Histogram(id, indexVar.asInstanceOf[InputVar], buckets)
      }
      // Multiplication
      case ProductNodeParseTree(id, multiplicands, _) => Product(id, multiplicands.map(constructSubTree(_)))
      // Addition
      case SumNodeParseTree(id, addends, _) => WeightedSum(id, addends.map{
        case (w, a) => WeightedAddend(constructSubTree(a), w)
      })
      // Reference to another node
      // We should not encounter any unresolved references if we have validated the parse tree before.
      case nr : NodeReferenceParseTree => constructSubTree(nr.declaration)
      // Input variables
      case InputVariableParseTree(id, index) => InputVar(id, index)
    })
  }


}
