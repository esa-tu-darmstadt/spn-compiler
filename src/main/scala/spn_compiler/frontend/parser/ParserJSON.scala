package spn_compiler.frontend.parser

// import java.io.File

import spn_compiler.graph_ir.nodes._
import play.api.libs.json._

/**
  * Entry point for parsing SPN from textual representation in input strings/files.
  */
object ParserJSON {

  private val nonLeafNodes : List[String] = List("Sum", "Product")

  /**
    * Parse an SPN from JSON representation in an IRGraph.
    *
    * @param jsonSpnObjStr Input JSON Object, representing a SPN (provided as [[String]]).
    * @return On success, returns an [[IRGraph]].
    */
  def parseJSON(jsonSpnObjStr: String): IRGraph = {
    // Parse input JSON-SPN-Object and retrieve its head, i.e.: root node
    // Assumption: There is exactly one node at the topmost level
    val json: JsObject = Json.parse(jsonSpnObjStr).asInstanceOf[JsObject]
    val spnRoot = constructSubTree(json)
    IRGraph(spnRoot, getInputList(json.values.head))
  }

  /**
    * Construct an IRNode, starting from the provided root node
    *
    * @param root [[JsValue]] representing the considered SPN subtree's root.
    * @return On success, returns an [[IRNode]].
    */
  def constructSubTree(root : JsValue) : IRNode = {
    // Since the cast to JsObject seems necessary in any case, use the 'fields' function to convert the JsValue
    // This yields a Seq(String, JsValue) representing the nodeType and its subtree.
    // Assumption: There is exactly one element in the sequence.
    val (nodeType : String, subtree : JsValue) = root.asInstanceOf[JsObject].fields.head
    val id : String = nodeType + "Node_" + subtree("id").as[Int]

    // Parent (i.e.: "non-leaf") nodes are handled below
    if (nonLeafNodes contains nodeType) {
      val subtreeChildren: JsValue = subtree("children")
      // Check if field 'children' exists (=> JsDefined) and if its type is of JsArray
      // If not: throw corresponding exceptions
      return subtreeChildren.result match {
        case _: JsDefined => subtreeChildren.validate[JsArray] match {

          // Handle the respective parent-node cases -> create their corresponding IRNode
          case s: JsSuccess[JsArray] => return nodeType match {
            case "Sum" => {
              // Create the list of addends out of the child-nodes, then create pairs, using the field 'weights'
              val addends = s.get.as[List[JsValue]].map(constructSubTree).zip(subtree("weights").as[List[Double]])
              WeightedSum(id, addends.map { case (a, w) => WeightedAddend(a, w) })
            }
            case "Product" => {
              Product(id, s.get.as[List[JsValue]].map(constructSubTree))
            }
          }

          case e: JsError => throw new RuntimeException(
            "Expected field type of 'children' is '%s' but was: '%s'\n%s"
              .format(JsArray.getClass, subtreeChildren.getClass, e.toString))
        }

        case _: JsUndefined => throw new RuntimeException(
          "Encountered '%s' non-leaf node without field 'children':\n%s"
            .format(nodeType, subtree.toString()))
      }
    }

    // Leaf nodes are handled below -> create their corresponding IRNode
    nodeType match {
      case "Categorical" => {
        System.err.println("WARNING: '%s' branch not yet implemented -> creating '%s'".format(nodeType, id + "_dummy"))
        InputVar(id + "_dummy", Int.MinValue)
      }
      case "Histogram" => {
        val indexVar : InputVar = getInputList(subtree).head
        val breaks = subtree("breaks").as[List[Int]]
        val densities = subtree("densities").as[List[Double]]
        require(breaks.size == (densities.size + 1), "Cannot construct histogram buckets from given breaks and values!")
        // Zip all elements but the last (init) with all elements but the first.
        // Result is a list of tuples with lower and upper bounds for each bucket.
        val listBreaks = breaks.init zip breaks.tail
        val buckets = (listBreaks zip densities).map { case ((lb, ub), d) => HistogramBucket(lb, ub, d) }
        Histogram(id, indexVar, buckets)
      }
      case "Poisson" => {
        val input: InputVar = getInputList(subtree).head
        val lambda = subtree("lambda").as[Double]
        PoissonDistribution(id, input, lambda)
      }
      case u => throw new RuntimeException("Unknown leaf-node type '%s'".format(u))
    }
  }

  /**
    * Retrieve a list of input variables, regarding the provided subtree.
    *
    * @param json JsValue representing the provided SPN subtree.
    * @return On success, returns a List of [[InputVar]].
    */
  def getInputList(json: JsValue) : List[InputVar] = {
    // "scope" field of the topmost JSON node represents an array of variables / inputs used by the SPN-subtree
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
    * Print the provided subtree to console.
    * (Used to train some Scala & PlayJSON with this method.)
    *
    * @param root [[JsValue]] representing the provided SPN subtree's root.
    */
  def printSubTree(root : JsValue, layer : Integer = 0) : Unit = {
    // Since the cast to JsObject seems necessary in any case, use the 'fields' function to convert the JsValue
    // This yields a Seq(String, JsValue) representing the nodeType and its subtree.
    val(nodeType : String, subtree : JsValue) = root.asInstanceOf[JsObject].fields.head
    println("  " * layer + nodeType)

    if (nonLeafNodes contains nodeType) {
      val subtreeValue: JsValue = subtree("children")
      subtreeValue.result match {
        case _: JsDefined => subtreeValue.validate[JsArray] match {
          case s: JsSuccess[JsArray] => s.get.as[List[JsValue]].foreach(printSubTree(_, layer + 1))
          case e: JsError => throw new RuntimeException(
            "Expected field type of 'children' is %s but was: %s\n%s"
              .format(JsArray.getClass, subtreeValue.getClass, e.toString))
        }
        case _: JsUndefined => throw new RuntimeException(
          "Encountered '%s' non-leaf node without children.\nSubtree: %s"
            .format(nodeType, subtree.toString()))
      }
    }
  }

}
