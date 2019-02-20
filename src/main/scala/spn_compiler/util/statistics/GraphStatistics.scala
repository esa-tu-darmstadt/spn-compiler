package spn_compiler.util.statistics

import java.io.{BufferedWriter, File, FileWriter}

import play.api.libs.functional.syntax._
import play.api.libs.json._
import spn_compiler.graph_ir.nodes._

import scala.collection.mutable


/**
  * Statistics tool to compute statistic numbers (e.g. number of sum-nodes) from an SPN.
  */
object GraphStatistics {

  /**
    * Compute statistics for given SPN.
    * @param spn SPN, represented as [[IRGraph]].
    */
  def computeStatistics(spn : IRGraph, statsFile : File): Unit = {
    val gs : GraphStatistics = computeSubtree(spn.rootNode)
    val json = Json.toJson(gs)
    val bw = new BufferedWriter(new FileWriter(statsFile))
    bw.write(Json.prettyPrint(json))
    bw.close()
  }

  private val computedNodes : mutable.Set[IRNode] = mutable.Set()

  private def computeSubtree(subtreeRoot : IRNode) : GraphStatistics = {
    if (computedNodes.contains(subtreeRoot))
      GraphStatistics()
    else {
      val result = subtreeRoot match {
        case iv : InputVar => GraphStatistics().addNode(iv)
        case h @ Histogram(_, indexVar, _) => GraphStatistics().addNode(h).merge(computeSubtree(indexVar))
        case p @ PoissonDistribution(_, inputVar, _) => GraphStatistics().addNode(p).merge(computeSubtree(inputVar))
        case ws @ WeightedSum(_, addends) =>
          addends.map(a => computeSubtree(a.addend)).fold(GraphStatistics().addNode(ws))((s1, s2)=> s1 merge s2)
        case p @ Product(_, multiplicands) =>
          multiplicands.map(m => computeSubtree(m)).fold(GraphStatistics().addNode(p))((s1, s2) => s1 merge s2)
        case _ => ??? /* Unexpected case */
      }
      computedNodes.add(subtreeRoot)
      result
    }
  }

  final case class GraphStatistics(numAdders : Int = 0, numMultipliers : Int = 0, numPoisson : Int = 0,
                                           numHistogram : Int = 0, numInputs : Int = 0,
                                           addOpStatistics : OperandStatistics = OperandStatistics(),
                                           mulOpStatistics : OperandStatistics = OperandStatistics()) {

    def addNode(node : IRNode) : GraphStatistics = node match {
      case iv : InputVar =>
        GraphStatistics(numAdders, numMultipliers, numPoisson, numHistogram, numInputs+1, addOpStatistics, mulOpStatistics)

      case h : Histogram =>
        GraphStatistics(numAdders, numMultipliers, numPoisson, numHistogram+1, numInputs, addOpStatistics, mulOpStatistics)

      case p : PoissonDistribution =>
        GraphStatistics(numAdders, numMultipliers, numPoisson+1, numHistogram, numInputs, addOpStatistics, mulOpStatistics)

      case WeightedSum(_, addends) =>
        GraphStatistics(numAdders+1, numMultipliers, numPoisson, numHistogram, numInputs,
          addOpStatistics.increment(addends.size), mulOpStatistics)

      case Product(_, multiplicands) =>
        GraphStatistics(numAdders, numMultipliers+1, numPoisson, numHistogram, numInputs,
          addOpStatistics, mulOpStatistics.increment(multiplicands.size))

      case _ => ??? /* Unexpected case */
    }

    def merge(gs : GraphStatistics) : GraphStatistics =
      GraphStatistics(numAdders+gs.numAdders, numMultipliers+gs.numMultipliers, numPoisson+gs.numPoisson,
        numHistogram+gs.numHistogram, numInputs+gs.numInputs, addOpStatistics.merge(gs.addOpStatistics),
        mulOpStatistics.merge(gs.mulOpStatistics))

    override def toString: String = {
      val sb : mutable.StringBuilder = new StringBuilder()
      sb.append("Number of input variables:\t%d\n".format(numInputs))
      sb.append("Number of weighted adders:\t%d\n".format(numAdders))
      sb.append("Number of multiplications:\t%d\n".format(numMultipliers))
      sb.append("Number of Poisson distr.:\t%d\n".format(numPoisson))
      sb.append("Number of histograms:\t\t%d\n".format(numHistogram))
      sb.append("Weighted adder number of operands - statistics:\n%s\n".format(addOpStatistics.toString))
      sb.append("Multiplier number of operands - statistics:\n%s\n".format(mulOpStatistics.toString))
      sb.toString()
    }
  }

  final case class OperandStatistics(histogram : Map[Int, Int] = Map()) {
    def increment(numOperands : Int) : OperandStatistics = {
      OperandStatistics(histogram + (numOperands -> (histogram.getOrElse(numOperands, 0)+1)))
    }

    def merge(os : OperandStatistics) : OperandStatistics = {
      OperandStatistics((histogram.keySet ++ os.histogram.keySet)
        .map(k => k -> (histogram.getOrElse(k, 0) + os.histogram.getOrElse(k, 0))).toMap)
    }

    override def toString: String =
      histogram.keySet.toList.sorted.map(k => "%d -> %d".format(k, histogram(k))).mkString("\n")

  }

  implicit val operandStatisticsWrite : Writes[OperandStatistics] = new Writes[OperandStatistics] {
    override def writes(o: OperandStatistics): JsValue = {
      var arr = Json.arr()
      for(k <- o.histogram.keySet){
        arr = arr :+ Json.obj("num_inputs" -> k, "appearances" -> o.histogram(k))
      }
      arr
    }
  }

  type OperandStatisticsEntry = (Int, Int)

  implicit val operandStatisticsRead : Reads[OperandStatisticsEntry] =
    ((JsPath \ "num_inputs").read[Int] and
      (JsPath \ "appearances").read[Int])((k, v) => k -> v)


  implicit val graphStatisticsWrite : Writes[GraphStatistics] = new Writes[GraphStatistics] {
    override def writes(o: GraphStatistics): JsValue = Json.obj(
      "numAdd" -> o.numAdders,
      "numMul" -> o.numMultipliers,
      "numPoisson" -> o.numPoisson,
      "numHist" -> o.numHistogram,
      "numInput" -> o.numInputs,
      "addStat" -> o.addOpStatistics,
      "mulStat" -> o.mulOpStatistics
    )
  }

  implicit val graphStatisticsRead : Reads[GraphStatistics] =
    ((JsPath \ "numAdd").read[Int] and
      (JsPath \ "numMul").read[Int] and
      (JsPath \ "numPoisson").read[Int] and
      (JsPath \ "numHist").read[Int] and
      (JsPath \ "numInput").read[Int] and
      (JsPath \ "addStat").read[Seq[OperandStatisticsEntry]] and
      (JsPath \ "mulStat").read[Seq[OperandStatisticsEntry]])((a, b, c, d, e, f, g) =>
      GraphStatistics(a, b, c, d, e, OperandStatistics(f.toMap), OperandStatistics(g.toMap)))

}

