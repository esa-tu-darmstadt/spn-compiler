package spn_compiler.frontend.parser

import fastparse.MultiLineWhitespace._
import fastparse._

import scala.io.Source

/**
  * Entry point for parsing SPN from textual representation in input strings/files.
  */
object Parser {

  /**
    * Parse an SPN from textual representation in an input string.
    * @param text Input string.
    * @return On success, returns a [[ParseTree]].
    */
  def parseString(text : String) : Unit = {
    // Parse input text.
    val parseResult = parse(text.trim, spn(_)) match {
      case Parsed.Success(parseTree, _) => parseTree
      case f : Parsed.Failure => throw new RuntimeException("Failed to parse SPN from input: \n"+f.trace().terminalsMsg)
    }
    // Perform identification on resulting parse-tree if the parser was successful.
    new Identification().performIdentification(parseResult)
    parseResult.validate()
  }

  /**
    * Parse an SPN from textual representation in a file.
    * @param fileName Input file name.
    * @return On success, returns a [[ParseTree]].
    */
  def parseFile(fileName : String) : Unit = parseString(Source.fromFile(fileName).mkString)

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
