package spn_compiler.graph_ir.analysis

import java.io.{BufferedWriter, File, FileWriter}

import play.api.libs.functional.syntax._
import play.api.libs.json._
import spn_compiler.graph_ir.nodes._

import scala.collection.mutable
import scala.io.Source

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

  /**
    * Read two graph statistics from JSON files, merge them and write back to a JSON file.
    * @param statsFile1 First JSON input file.
    * @param statsFile2 Second JSON input file.
    * @param outFile Output JSON file.
    */
  def mergeStatistics(statsFile1 : File, statsFile2 : File, outFile : File): Unit ={
    val json1 = Source.fromFile(statsFile1).mkString
    val gs1 = Json.parse(json1).as[GraphStatistics]
    val json2 = Source.fromFile(statsFile2).mkString
    val gs2 = Json.parse(json2).as[GraphStatistics]
    val json = Json.toJson(gs1 merge gs2)
    val bw = new BufferedWriter(new FileWriter(outFile))
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

  final def countAddOperands(node : IRNode) : Option[Int] = node match {
    case WeightedSum(_, addends) => Some(addends.size)
    case _ => None
  }

  final def countMulOperands(node : IRNode) : Option[Int] = node match {
    case Product(_, multiplicands) => Some(multiplicands.size)
    case _ => None
  }

  /**
    * Object representation for SPN graph statistics.
    * @param operatorStatistics [[OperatorStatistics]] about the arithmetic operators in the SPN.
    * @param addOpStatistics [[OperandStatistics]] about the adders in the SPN.
    * @param mulOpStatistics [[OperandStatistics]] about the multiplications in the SPN.
    */
  final case class GraphStatistics(operatorStatistics: OperatorStatistics = OperatorStatistics(),
                                           addOpStatistics : OperandStatistics = OperandStatistics()(countAddOperands),
                                           mulOpStatistics : OperandStatistics = OperandStatistics()(countMulOperands)) {

    def addNode(node : IRNode) : GraphStatistics = GraphStatistics(operatorStatistics.addNode(node),
      addOpStatistics.addNode(node), mulOpStatistics.addNode(node))

    def merge(gs : GraphStatistics) : GraphStatistics =
      GraphStatistics(operatorStatistics.merge(gs.operatorStatistics), addOpStatistics.merge(gs.addOpStatistics),
        mulOpStatistics.merge(gs.mulOpStatistics))

    override def toString: String = {
      val sb : mutable.StringBuilder = new StringBuilder()
      sb.append(operatorStatistics.toString)
      sb.append("Weighted adder number of operands - statistics:\n%s\n".format(addOpStatistics.toString))
      sb.append("Multiplier number of operands - statistics:\n%s\n".format(mulOpStatistics.toString))
      sb.toString()
    }
  }

  sealed trait Statistics[T <: Statistics[T]] {
    def addNode(node : IRNode) : T
    def merge(stat : T) : T
  }

  final case class OperatorStatistics(numAdders : Int = 0, numMultipliers : Int = 0, numPoisson : Int = 0,
                                      numHistogram : Int = 0, numInputs : Int = 0)
    extends Statistics[OperatorStatistics]{
    def addNode(node : IRNode) : OperatorStatistics = node match {
      case iv : InputVar =>
        OperatorStatistics(numAdders, numMultipliers, numPoisson, numHistogram, numInputs+1)

      case h : Histogram =>
        OperatorStatistics(numAdders, numMultipliers, numPoisson, numHistogram+1, numInputs)

      case p : PoissonDistribution =>
        OperatorStatistics(numAdders, numMultipliers, numPoisson+1, numHistogram, numInputs)

      case WeightedSum(_, addends) =>
        OperatorStatistics(numAdders+1, numMultipliers, numPoisson, numHistogram, numInputs)

      case Product(_, multiplicands) =>
        OperatorStatistics(numAdders, numMultipliers+1, numPoisson, numHistogram, numInputs)

      case _ => ??? /* Unexpected case */
    }

    def merge(gs : OperatorStatistics) : OperatorStatistics =
      OperatorStatistics(numAdders+gs.numAdders, numMultipliers+gs.numMultipliers, numPoisson+gs.numPoisson,
        numHistogram+gs.numHistogram, numInputs+gs.numInputs)

    override def toString: String = {
      val sb : mutable.StringBuilder = new StringBuilder()
      sb.append("Number of input variables:\t%d\n".format(numInputs))
      sb.append("Number of weighted adders:\t%d\n".format(numAdders))
      sb.append("Number of multiplications:\t%d\n".format(numMultipliers))
      sb.append("Number of Poisson distr.:\t%d\n".format(numPoisson))
      sb.append("Number of histograms:\t\t%d\n".format(numHistogram))
      sb.toString()
    }
  }

  /**
    * Object representation of the operand statistics.
    * @param histogram Histogram reflecting the number of operator instances per
    *                  number of operands.
    */
  final case class OperandStatistics(histogram : Map[Int, Int] = Map()) (lambda : IRNode => Option[Int])
    extends Statistics[OperandStatistics] {

    override def addNode(node: IRNode): OperandStatistics = increment(lambda(node))

    private def increment(numOperands : Option[Int]) : OperandStatistics =
      if(numOperands.isDefined)
        OperandStatistics(histogram + (numOperands.get -> (histogram.getOrElse(numOperands.get, 0)+1)))(lambda)
      else
        this


    override def merge(stat: OperandStatistics): OperandStatistics = {
      OperandStatistics((histogram.keySet ++ stat.histogram.keySet)
        .map(k => k -> (histogram.getOrElse(k, 0) + stat.histogram.getOrElse(k, 0))).toMap)(lambda)
    }

    override def toString: String =
      histogram.keySet.toList.sorted.map(k => "%d -> %d".format(k, histogram(k))).mkString("\n")

  }

  //
  // Interface to the Play Scala JSON library.
  // Defines (de-)serialization of the graph- and operand statistics.
  //
  implicit val operatorStatisticsWrite : Writes[OperatorStatistics] = new Writes[OperatorStatistics] {
    override def writes(o: OperatorStatistics): JsValue = {
      Json.obj("numAdd" -> o.numAdders,
        "numMul" -> o.numMultipliers,
        "numPoisson" -> o.numPoisson,
        "numHist" -> o.numHistogram,
        "numInput" -> o.numInputs)
    }
  }

  implicit val operatorStatisticsRead : Reads[OperatorStatistics] =
    ((JsPath \ "numAdd").read[Int] and
      (JsPath \ "numMul").read[Int] and
      (JsPath \ "numPoisson").read[Int] and
      (JsPath \ "numHist").read[Int] and
      (JsPath \ "numInput").read[Int])((a, m, p, h, i) => OperatorStatistics(a, m, p, h, i))

  implicit val operandStatisticsWrite : Writes[OperandStatistics] = new Writes[OperandStatistics] {
    override def writes(o: OperandStatistics): JsValue = {
      var arr = Json.arr()
      for(k <- o.histogram.keySet.toList.sorted){
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
      "opStat" -> o.operatorStatistics,
      "addStat" -> o.addOpStatistics,
      "mulStat" -> o.mulOpStatistics
    )
  }

  implicit val graphStatisticsRead : Reads[GraphStatistics] =
    ((JsPath \ "opStat").read[OperatorStatistics] and
      (JsPath \ "addStat").read[Seq[OperandStatisticsEntry]] and
      (JsPath \ "mulStat").read[Seq[OperandStatisticsEntry]])((os, as, ms) =>
      GraphStatistics(os, OperandStatistics(as.toMap)(countAddOperands),
        OperandStatistics(ms.toMap)(countMulOperands)))

}
