package spn_compiler.graph_ir.transform

import spn_compiler.graph_ir.nodes._

object Marginalization {

  def constructMarginalizedGraph(spn : IRGraph, marginals : Set[InputVar]) : IRGraph =
    IRGraph(constructMarginalizedSubgraph(spn.rootNode, marginals), spn.inputVariables)

  def constructMarginalizedSubgraph(rootNode: IRNode, marginals: Set[InputVar]): IRNode = rootNode match {
    case pd @ PoissonDistribution(id, variable, lambda) =>
      if(marginals.contains(variable))
        Marginal(id)
      else
        pd

    case h @ Histogram(id, indexVar, buckets) =>
      if(marginals.contains(indexVar))
        Marginal(id)
      else
        h

    case WeightedAddend(addend, weight) =>
      WeightedAddend(constructMarginalizedSubgraph(addend, marginals), weight)

    case WeightedSum(id, addends) =>
      WeightedSum(id, addends.map(constructMarginalizedSubgraph(_, marginals).asInstanceOf[WeightedAddend]))

    case Sum(id, addends) =>
      Sum(id, addends.map(constructMarginalizedSubgraph(_, marginals)))

    case Product(id, multiplicands) =>
      val nonMarginals = multiplicands.map(constructMarginalizedSubgraph(_, marginals)).filter(!_.isInstanceOf[Marginal])
      nonMarginals match {
        case Nil => Marginal(id)
        case head :: Nil => head
        case _ => Product(id, nonMarginals)
      }

    case _ => throw new MatchError("Should not reach default case")
  }


}
