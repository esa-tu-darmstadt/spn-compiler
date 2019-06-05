package spn_compiler.backend.software.cpu.ast_generation.serial

import spn_compiler.backend.software.ast.construct._
import spn_compiler.backend.software.ast.nodes.function.{ASTFunction, ASTFunctionParameter}
import spn_compiler.backend.software.ast.nodes.module.ASTModule
import spn_compiler.backend.software.ast.nodes.types.{IntegerType, RealType, StructType, VoidType}
import spn_compiler.backend.software.ast.nodes.value.ASTValue
import spn_compiler.graph_ir.nodes._

import scala.collection.mutable

class ASTGeneration {

  def createAST(graph : IRGraph) : ASTModule = {
    val module = new ASTModule("spn")
    val inputStructType = module.createStructType("activation",
      graph.inputVariables.map(v => (s"input_${v.id}", IntegerType)):_*)
    val spnFunction = createSPNFunction(graph.rootNode, module, inputStructType)
    val toplevelFunction = createTopLevelFunction(spnFunction, module, inputStructType)
    module
  }

  protected def createTopLevelFunction(spnFunction : ASTFunction, module : ASTModule, inputStructType : StructType) : ASTFunction = {
    val numElements = module.createFunctionParameter("num_examples", IntegerType)
    val inputData = module.createFunctionParameter("input_data", module.createArrayType(inputStructType))
    val outputData = module.createFunctionParameter("output_data", module.createArrayType(RealType))
    val topLevelFunction = module.defineLocalFunction("spn_toplevel", VoidType, numElements, inputData, outputData)
    module.setInsertionPoint(topLevelFunction.body)
    val loopVar = module.createVariable(IntegerType, "i")
    module.insertStatement(module.declareVariable(loopVar))
    val constantZero = module.constantValue(IntegerType, 0)
    val forLoop = module.insertStatement(module.forLoop(loopVar, constantZero,
      numElements, module.constantValue(IntegerType, 1)))
    module.setInsertionPoint(forLoop.body)
    module.insertStatement(module.assignIndex(outputData, loopVar,
      module.call(spnFunction, module.readIndex(inputData, loopVar))))
    topLevelFunction
  }

  protected def createSPNFunction(spnRoot : IRNode, module : ASTModule, inputStructType : StructType) : ASTFunction = {
    val inputParam = module.createFunctionParameter("activation", inputStructType)
    val spnFunction = module.defineLocalFunction("spn", RealType, inputParam)
    module.setInsertionPoint(spnFunction.body)
    module.insertStatement(module.ret(constructSubAST(spnRoot, module, inputParam)))
    spnFunction
  }

  private val constructedSubGraphs : mutable.Map[IRNode, ASTValue] = mutable.Map()

  protected def constructSubAST(subTreeRot : IRNode, module : ASTModule, inputParam : ASTFunctionParameter) : ASTValue =
    constructedSubGraphs.getOrElseUpdate(subTreeRot, subTreeRot match {
      case InputVar(id, _) => module.readElement(module.referenceVariable(inputParam), s"input_$id")

      case Histogram(id, indexVar, buckets) => {
        val arrayInit = module.initArray(RealType, buckets.flatMap(b => (b.lowerBound until b.upperBound).map(_ => b.value)):_*)
        val globalVar = module.createVariable(module.createArrayType(RealType), id)
        module.declareGlobalVariable(globalVar, arrayInit)
        val activation = constructSubAST(indexVar, module, inputParam)
        module.readIndex(module.referenceVariable(globalVar), activation)
      }

      case WeightedSum(id, addends) => {
        val operands = addends.map(wa => module.mul(constructSubAST(wa.addend, module, inputParam), module.constantValue(RealType, wa.weight)))
        operands.tail.fold[ASTValue](operands.head)(module.add)
      }

      case Sum(id, addends) => {
        val operands = addends.map(constructSubAST(_, module, inputParam))
        operands.tail.fold(operands.head)(module.add)
      }

      case Product(id, multiplicands) => {
        val operands = multiplicands.map(constructSubAST(_, module, inputParam))
        operands.tail.fold(operands.head)(module.mul)
      }
    })

}
