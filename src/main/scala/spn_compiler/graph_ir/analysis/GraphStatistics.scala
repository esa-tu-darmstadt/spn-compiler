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

  private val computedNodes : mutable.Map[IRNode, GraphStatistics] = mutable.Map()

  private def computeSubtree(subtreeRoot : IRNode) : GraphStatistics = {
    if (computedNodes.contains(subtreeRoot))
      computedNodes(subtreeRoot)
    else {
      val result = subtreeRoot match {
        case iv : InputVar => GraphStatistics().addNode(iv)
        case h @ Histogram(_, indexVar, _) => {
          val subtreeResult = computeSubtree(indexVar)
          GraphStatistics().addNode(h).merge(subtreeResult)
        }
        case p @ PoissonDistribution(_, inputVar, _) => {
          val subtreeResult = computeSubtree(inputVar)
          GraphStatistics().addNode(p).merge(subtreeResult)
        }
        case ws @ WeightedSum(_, addends) =>{
          val operandResults = addends.map(op => computeSubtree(op.addend))
          operandResults.fold(GraphStatistics())((s1, s2)=> s1 merge s2).addNode(ws)
        }
        case p @ Product(_, multiplicands) =>
          val operandResults = multiplicands.map(computeSubtree)
          operandResults.fold(GraphStatistics())((s1, s2) => s1 merge s2).addNode(p)
        case _ => ??? /* Unexpected case */
      }
      computedNodes += subtreeRoot -> result
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
                                           mulOpStatistics : OperandStatistics = OperandStatistics()(countMulOperands),
                                   numericStatistics: NumericStatistics = NumericStatistics()) {

    def addNode(node : IRNode) : GraphStatistics = GraphStatistics(operatorStatistics.addNode(node),
      addOpStatistics.addNode(node), mulOpStatistics.addNode(node), numericStatistics.addNode(node))

    def merge(gs : GraphStatistics) : GraphStatistics =
      GraphStatistics(operatorStatistics.merge(gs.operatorStatistics), addOpStatistics.merge(gs.addOpStatistics),
        mulOpStatistics.merge(gs.mulOpStatistics), numericStatistics.merge(gs.numericStatistics))

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

  type DynamicRange = (Double, Double)

  final case class NumericStatistics(minValue : Double = 1.0, maxValue : Double = 0.0,
                                     ranges : Map[IRNode, DynamicRange] = Map()) extends Statistics[NumericStatistics]{

    def addNode(node : IRNode) : NumericStatistics = node match {
      case h @ Histogram(_, _, buckets) => {
        val bucketValues = buckets.map(_.value)
        val minimum = bucketValues.fold(minValue)(min)
        val maximum = bucketValues.fold(maxValue)(max)
        NumericStatistics(minimum, maximum, ranges + (h -> (minimum, maximum)))
      }

      case p : PoissonDistribution => this // TODO Compute min. and max. values for Poisson distribution.

      case ws @ WeightedSum(_, addends) => {
        val minOperands = addends.map(a => ranges(a.addend)._1)
        val maxOperands = addends.map(a => ranges(a.addend)._2)
        val minSum = (minOperands zip addends.map(_.weight)).map{case(v, w) => v*w}.fold(0.0)(_+_)
        val maxSum = (maxOperands zip addends.map(_.weight)).map{case(v, w) => v*w}.fold(0.0)(_+_)
        val minimum = minOperands.fold(min(minValue, minSum))(min)
        val maximum = maxOperands.fold(max(maxValue, maxSum))(max)
        NumericStatistics(minimum, maximum, ranges + (ws -> (minimum, maximum)))
      }

      case p @ Product(_, multiplicands) => {
        val minOperands = multiplicands.map(ranges(_)._1)
        val maxOperands = multiplicands.map(ranges(_)._2)
        val minProd = minOperands.fold(1.0)(_*_)
        val maxProd = maxOperands.fold(1.0)(_*_)
        val minimum = minOperands.fold(min(minValue, minProd))(min)
        val maximum = maxOperands.fold(max(maxValue, maxProd))(max)
        NumericStatistics(minimum, maximum, ranges + (p -> (minimum, maximum)))
      }

      case iv : InputVar => NumericStatistics()

      case _ => ??? /* Unexpected case */
    }

    def merge(gs : NumericStatistics) : NumericStatistics =
      NumericStatistics(min(minValue, gs.minValue), max(maxValue, gs.maxValue), ranges++gs.ranges)

    private def min(a : Double, b : Double) : Double = (a, b) match {
      case(0.0,0.0) => 0.0
      case(0.0, v) => v
      case(v, 0.0) => v
      case(v1, v2) => Math.min(v1, v2)
    }

    private def max(a : Double, b : Double) : Double = (a,b) match {
      case(1.0, 1.0) => 1.0
      case(1.0, v) => v
      case(v, 1.0) => v
      case(v1, v2) => Math.max(v1, v2)
    }

    override def toString: String = {
      val sb : mutable.StringBuilder = new StringBuilder()
      sb.append("Smallest non-zero value:\t%f\n".format(minValue))
      sb.append("Biggest non-one value:\t%f\n".format(maxValue))
      sb.toString()
    }
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

  implicit val numericStatisticsWrite : Writes[NumericStatistics] = new Writes[NumericStatistics] {
    override def writes(o: NumericStatistics): JsValue =
      Json.obj("minimum" -> o.minValue, "maximum" -> o.maxValue)
  }

  implicit val numericStatisticsRead : Reads[NumericStatistics] =
    ((JsPath \ "minimum").read[Double] and
      (JsPath \ "maximum").read[Double])((min, max) => NumericStatistics(min, max))


  implicit val graphStatisticsWrite : Writes[GraphStatistics] = new Writes[GraphStatistics] {
    override def writes(o: GraphStatistics): JsValue = Json.obj(
      "opStat" -> o.operatorStatistics,
      "addStat" -> o.addOpStatistics,
      "mulStat" -> o.mulOpStatistics,
      "numStat" -> o.numericStatistics
    )
  }

  implicit val graphStatisticsRead : Reads[GraphStatistics] =
    ((JsPath \ "opStat").read[OperatorStatistics] and
      (JsPath \ "addStat").read[Seq[OperandStatisticsEntry]] and
      (JsPath \ "mulStat").read[Seq[OperandStatisticsEntry]] and
      (JsPath \ "numStat").read[NumericStatistics])((os, as, ms, ns) =>
      GraphStatistics(os, OperandStatistics(as.toMap)(countAddOperands),
        OperandStatistics(ms.toMap)(countMulOperands), ns))

}
