package spn_compiler.frontend.parser

// import java.io.File

import fastparse.MultiLineWhitespace._
import fastparse._
import spn_compiler.graph_ir.nodes
import spn_compiler.graph_ir.nodes.{IRGraph, InputVar}
import play.api.libs.json._
import play.api.libs.functional.syntax._

import scala.io.Source

/**
  * Entry point for parsing SPN from textual representation in input strings/files.
  */
object ParserJSON {

  private val nonLeafNodes : List[String] = List("Sum", "Product")

  /**
    * Parse an SPN from textual representation in an input string.
    *
    * @param jsonSpnObjStr Input JSON Object, representing a SPN (provided as string).
    * @return On success, returns a [[ParseTree]].
    */
  def parseJSON(jsonSpnObjStr: String): String = {
    // Parse input JSON-SPN-Object and retrieve its head, i.e.: root node
    // Assumption: There is exactly one node at the topmost level
    val json: JsObject = Json.parse(jsonSpnObjStr).asInstanceOf[JsObject]

    println("__Got  : " + getInputList(json.values.head))
    println("__Class: " + getInputList(json.values.head).getClass)

    println(json.fields.head._1)

    val children_layer_1 = json.values.head("children").validate[JsArray].get.as[List[JsValue]]
    printChildren(children_layer_1, 1)

    // implicit val irGraphReads = (
    // ) (IRGraph)

    /*
    val parseResult = parse(text.trim, spn(_)) match {
      case Parsed.Success(parseTree, _) => parseTree
      case f : Parsed.Failure => throw new RuntimeException("Failed to parse SPN from input: \n"+f.trace().terminalsMsg)
    }
    // Perform identification on resulting parse-tree if the parser was successful.
    new Identification().performIdentification(parseResult)
    parseResult.validate()
    new IRConstruction(parseResult).constructIRGraph
    */

    // TODO: Return useful info -> ParseTree ?
    "Reached end of parseJSON"
  }

  def printChildren(childList : List[JsValue], layer : Integer = 0) : Unit = {
    for (child <- childList) {
      // Since the cast to JsObject seems necessary in any case, use the 'fields' function to convert the JsValue
      // This yields a Seq(String, JsValue) representing the nodeType and its subtree.
      for (field <- child.asInstanceOf[JsObject].fields) {
        val nodeType : String = field._1
        val subtree : JsValue = field._2
        println("  " * layer + nodeType)

        if (nonLeafNodes contains nodeType) {
          val subtreeValue: JsValue = subtree("children")
          subtreeValue.result match {
            case _ : JsDefined => subtreeValue.validate[JsArray] match {
              case s: JsSuccess[JsArray] => printChildren(s.get.as[List[JsValue]], layer + 1)
              case e: JsError => throw new RuntimeException(
                "Expected field type of 'children' is %s but was: %s\n%s"
                  .format(JsArray.getClass, subtreeValue.getClass, e.toString))
            }
            case _ : JsUndefined => throw new RuntimeException(
              "Encountered '%s' non-leaf node without children.\nSubtree: %s"
                .format(nodeType, subtree.toString()))
          }
        }
      }
    }
  }

  /**
    * Retrieve a list of input variables, regarding the provided subtree.
    *
    * @param json JsValue representing the provided SPN subtree.
    * @return On success, returns a List of [[InputVar]].
    */
  def getInputList(json: JsValue) : List[InputVar] = {
    // "scope" of topmost JSON node equals an array of variables / inputs used by the SPN-subtree
    json("scope").validate[JsArray] match {
      case s: JsSuccess[JsArray] => {
        // Inputs may be JsNumber and JsString in this case, contained by the JsArray
        // Additionally, the JsStrings contain " characters, which have to be removed
        // Goal: retrieve a List[String] without " in the input variable names
        val vars : List[String] = s.get.as[List[JsValue]].map(e => e.toString().filterNot(c => Set('\"').contains(c)))

        // Now replace each variable name with its respective InputVar, which yields: List[InputVar]
        vars.map(v => InputVar(v, vars.indexOf(v)))
      }
      case e: JsError => throw new RuntimeException("Failed to retrieve input variables from JSON:\n" + e.toString)
    }
  }

  /**
    * Parse an SPN from textual representation in a file.
    * @param file Input file name.
    * @return On success, returns a [[ParseTree]].
    */
  // def parseFile(file : File) : IRGraph = parseJSON(Source.fromFile(file).mkString)

  /*
   * Terminals
   *
   * ID	- Arbitrary string, starting with a letter
   * REAL	- Real number, potentially in scientific notation
   * INT	- Integer number, must not start with '0'
   *
   */
  private def digits[ _ : P]    = P( CharsWhileIn("0-9") )
  private def sign[_ : P]   = P( CharIn("+\\-") )
  private def exponent[_ : P]   = P( CharIn("eE") ~ sign.? ~ digits )
  private def fractional[_ : P] = P( "." ~ digits )
  private def integral [_ : P]    = P( "0" | CharIn("1-9") ~ digits.? )
  // Integer numbers
  private def integer [_ : P]   = P( sign.? ~ integral ).!.map(_.toInt)
  // Real numbers, potentially in scientific notation
  private def real [_ : P]    = P( sign.? ~ integral ~ fractional ~ exponent.?).!.map(_.toDouble)
  // All identifying strings starting with a letter. Strings may only contain letters, digits and underscores.
  private def id [_ : P]    = P( CharsWhile(_.isLetter) ~ CharsWhile(c => c.isLetterOrDigit | c == '_').? ).!

  /*
   * Productions
   */

  // spn := node inputs
  private def spn [_ : P] = P( node ~ inputs).map{case(r, inputs) => ParseTree(r, inputs)}

  // node := sumNode | productNode | histogramNode | poissonNode
  private def node [_ : P] : P[ParseTreeNode] = sumNode | productNode | histogramNode | poissonNode

  // weightedOp := REAL '*' ID
  private def weightedOp [_ : P] =
  P( real ~ "*" ~ id ).map{case(d, r) => (d, NodeReferenceParseTree(r))}

  // sumNode := ID 'SumNode' '(' weightedOp (',' weightedOp)* ')' '{' node* '}'
  private def sumNode [_ : P] =
  P( id ~ "SumNode" ~/ "(" ~ weightedOp.rep(sep = ","./) ~ ")" ~ "{" ~ node.rep./ ~ "}" )
      .map{case(i, ops, nodes) => SumNodeParseTree(i, ops.toList, nodes.toList)}

  // productNode := ID 'ProductNode' '(' ID (',' ID)* ')' '{' node* '}'
  private def productNode [_ : P] =
  P( id ~ "ProductNode" ~/ "(" ~ id.rep(sep = ","./) ~ ")" ~ "{" ~ node.rep./ ~ "}" )
      .map{case(i, ops, nodes) => ProductNodeParseTree(i, ops.toList.map(r => NodeReferenceParseTree(r)), nodes.toList)}

  // histogramBreak := INT '.'
  private def histogramBreak [_ : P] = P( integer ~ "." )

  // histogramNode := ID 'Histogram' '(' ID '|' '[' histogramBreak (',' histogramBreak)* ']' ';' '[' REAL (',' REAL)* ']' ')'
  private def histogramNode [_ : P] =
  P( id ~ "Histogram" ~/ "(" ~ id ~ "|" ~ "[" ~ histogramBreak.rep(sep = ","./) ~ "]" ~ ";" ~ "[" ~ real.rep(sep = ","./) ~ "]" ~ ")" )
      .map{case(i, v, breaks, values) => HistogramNodeParseTree(i, NodeReferenceParseTree(v), breaks.toList, values.toList)}

  // poissonNode := ID 'P' '(' ID '|' 'lambda' '=' REAL ')'
  private def poissonNode [_ : P] =
  P( id ~ "P" ~/ "(" ~ id ~ "|" ~ "lambda" ~ "=" ~ real ~ ")" )
      .map{case(i, v, l) => PoissonNodeParseTree(i, NodeReferenceParseTree(v), l)}

  // inputs := '#' ID (';' ID)*
  private def inputs [_ : P] =
    P( "#" ~ id.rep(sep = ";")).map(l => l.toList.map(v => InputVariableParseTree(v, l.indexOf(v))))



}
