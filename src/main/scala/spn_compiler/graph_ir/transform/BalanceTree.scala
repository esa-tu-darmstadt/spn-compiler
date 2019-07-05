package spn_compiler.graph_ir.transform

import spn_compiler.graph_ir.nodes._

object BalanceTree {

  def balanceTree(graph : IRGraph) : IRGraph = IRGraph(balanceSubtree(graph.rootNode), graph.inputVariables)

  private def balanceSubtree(rootNode: IRNode): IRNode = rootNode match {
    case pd : PoissonDistribution => pd
    case h : Histogram => h
    case m : Marginal => m
    case WeightedSum(id, addends) => splitWeightedSum(addends, id)
    case Sum(id, addends) => splitSum(addends, id)
    case Product(id, multiplicands) => splitProduct(multiplicands, id)
    case _ => throw new MatchError("Recursion should not reach default case")
  }

  private def splitWeightedSum(addends : List[WeightedAddend], baseID : String) : IRNode = addends match {
    case left :: right :: Nil =>
      // Exactly two addends, recurse on them and construct new weighted sum with same weights.
      WeightedSum(baseID, List(WeightedAddend(balanceSubtree(left.addend), left.weight),
        WeightedAddend(balanceSubtree(right.addend), right.weight)))

    case left1 :: left2 :: right :: Nil =>
      // Exactly three addends, recurse on the two left and construct a new weighted sum with
      // 1.0 as left weight and the orignal weight on the right side.
      val leftNew = splitWeightedSum(List(left1, left2), s"${baseID}_left")
      val rightNew = balanceSubtree(right.addend)
      WeightedSum(baseID, List(WeightedAddend(leftNew, 1.0), WeightedAddend(rightNew, right.weight)))

    case l @ List(_*) if l.size >= 4 =>
      // At least 4 addends. Split them in two halves, recurse on each half and
      // construct a new weighted sum with 1.0 as weight on both sides.
      val (leftHalf, rightHalf) = l.splitAt(l.size/2)
      val leftNew = splitWeightedSum(leftHalf, s"${baseID}_left")
      val rightNew = splitWeightedSum(rightHalf, s"${baseID}_right")
      WeightedSum(baseID, List(WeightedAddend(leftNew, 1.0), WeightedAddend(rightNew, 1.0)))

    case _ => throw new MatchError("Should not reach default case")
  }

  private def splitSum(addends : List[IRNode], baseID : String) : IRNode = addends match {
    case a :: Nil =>
      balanceSubtree(a)

    case left :: right :: Nil =>
      Sum(baseID, List(balanceSubtree(left), balanceSubtree(right)))

    case l @ List(_*) if l.size >= 3 =>
      val (leftHalf, rightHalf) = l.splitAt(l.size/2)
      Sum(baseID, List(splitSum(leftHalf, s"${baseID}_left"), splitSum(rightHalf, s"${baseID}_right")))

    case _ => throw new MatchError("Recursion should not reach default case")
  }

  private def splitProduct(multiplicands : List[IRNode], baseID : String) : IRNode = multiplicands match {
    case a :: Nil =>
      balanceSubtree(a)

    case left :: right :: Nil =>
      Product(baseID, List(balanceSubtree(left), balanceSubtree(right)))

    case l @ List(_*) if l.size >= 3 =>
      val (leftHalf, rightHalf) = l.splitAt(l.size/2)
      Product(baseID, List(splitProduct(leftHalf, s"${baseID}_left"),
        splitProduct(rightHalf, s"${baseID}_right")))

    case _ => throw new MatchError("Recursion should not reach default case")
  }
}
