package spn_compiler.backend.software.cpu.ast_generation.serial

import spn_compiler.backend.software.ast.construct._
import spn_compiler.backend.software.ast.extensions.simulation.lns.LNS
import spn_compiler.backend.software.ast.nodes.function.{ASTFunction, ASTFunctionParameter}
import spn_compiler.backend.software.ast.nodes.module.ASTModule
import spn_compiler.backend.software.ast.nodes.types._
import spn_compiler.backend.software.ast.nodes.value.ASTValue
import spn_compiler.backend.software.ast.predef._
import spn_compiler.driver.config.CPPCompileConfig
import spn_compiler.driver.config.LNS_SW_Type._
import spn_compiler.graph_ir.nodes._

import scala.collection.mutable

class ASTGeneration[C <: CPPCompileConfig[C]](private val config : C) {

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
    if(config.isLNSSimulationEnabled){
      // Initialize interpolator if LNS simulation is enabled
      module.insertStatement(module.createCallStatement(LNSInit,
        module.constantValue(IntegerType, config.lnsIntegerBits),
        module.constantValue(IntegerType, config.lnsFractionBits),
        module.constantValue(RealType, config.lnsInterpolationError)))
    }
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
    calculateGraphDepth(spnRoot)
    val result = constructSubAST(spnRoot, module, inputParam)
    // Convert result to double if LNS simulation is enabled
    module.insertStatement(module.ret(if(config.isLNSSimulationEnabled) lns2Double(result, module) else result))
    spnFunction
  }

  private val constructedSubGraphs : mutable.Map[IRNode, ASTValue] = mutable.Map()

  protected def constructSubAST(subTreeRot : IRNode, module : ASTModule, inputParam : ASTFunctionParameter) : ASTValue =
    constructedSubGraphs.getOrElseUpdate(subTreeRot, subTreeRot match {
      case InputVar(id, _) => module.readElement(module.referenceVariable(inputParam), s"input_$id")

      case h @ Histogram(id, indexVar, buckets) => {
        val arrayInit =
          if(config.isLNSSimulationEnabled){
            module.initArray(buckets.flatMap(b => (b.lowerBound until b.upperBound).map(_ => double2LNS(b.value, module))):_*)
          }
          else if(config.isLNSSoftwareSimulationEnabled) {
            module.initArray(buckets.flatMap(b => (b.lowerBound until b.upperBound).map(_ => double2LNSSW(b.value, module))):_*)
          }
          else if(config.isPositSimulationEnabled) {
            module.initArray(buckets.flatMap(b => (b.lowerBound until b.upperBound).map(_ => double2Posit(b.value, module))):_*)
          }
          else if(config.isFPSimulationEnabled) {
            module.initArray(buckets.flatMap(b => (b.lowerBound until b.upperBound).map(_ => double2FPSim(b.value, module))):_*)
          }
          else if(config.isFixedPointSimulationEnabled) {
            module.initArray(buckets.flatMap(b => (b.lowerBound until b.upperBound).map(_ => double2Fixed(b.value, module))):_*)
          }
          else {
            module.initArray(RealType, buckets.flatMap(b => (b.lowerBound until b.upperBound).map(_ => b.value)):_*)
          }
        val arrayElementType =
          if(config.isLNSSimulationEnabled){
            LNSType
          }
          else if(config.isLNSSoftwareSimulationEnabled) {
            if(config.lnsSoftwareType == FLOAT)
              LNSSWSimTypeFloat
            else
              LNSSWSimTypeDouble
          }
          else if(config.isPositSimulationEnabled) {
            PositType
          }
          else if(config.isFPSimulationEnabled) {
            FPSimType
          }
          else if(config.isFixedPointSimulationEnabled){
            FixedType
          }
          else {
            RealType
          }
        val globalVar =
          module.createVariable(module.createArrayType(arrayElementType), id)
        module.declareGlobalVariable(globalVar, arrayInit)
        val activation = constructSubAST(indexVar, module, inputParam)
        module.readIndex(module.referenceVariable(globalVar), activation)
          .addAnnotation("spn.graph.depth", graphDepth.getOrElse(h, 0))
      }

      case ws @ WeightedSum(id, addends) => {
        val weights =
          if(config.isLNSSimulationEnabled){
            addends.map(wa => double2LNS(wa.weight, module))
          }
          else if(config.isLNSSoftwareSimulationEnabled){
            addends.map(wa => double2LNSSW(wa.weight, module))
          }
          else if(config.isPositSimulationEnabled){
            addends.map(wa => double2Posit(wa.weight, module))
          }
          else if(config.isFPSimulationEnabled){
            addends.map(wa => double2FPSim(wa.weight, module))
          }
          else if(config.isFixedPointSimulationEnabled){
            addends.map(wa => double2Fixed(wa.weight, module))
          }
          else {
            addends.map(wa => module.constantValue(RealType, wa.weight))
          }
        val adds = addends.map(wa => constructSubAST(wa.addend, module, inputParam))

        val operands = (adds zip weights).map{case(a, w) => module.mul(a, w)}
        val rhs = operands.tail.fold[ASTValue](operands.head)(module.add)
        val variable = module.createVariable(rhs.getType, id)
        module.insertStatement(module.declareVariable(variable, rhs))
        module.readVariable(variable).addAnnotation("spn.graph.depth", graphDepth.getOrElse(ws, 0))
      }

      case s @ Sum(id, addends) => {
        val operands = addends.map(constructSubAST(_, module, inputParam))
        val rhs = operands.tail.fold(operands.head)(module.add)
        val variable = module.createVariable(rhs.getType, id)
        module.insertStatement(module.declareVariable(variable, rhs))
        module.readVariable(variable).addAnnotation("spn.graph.depth", graphDepth.getOrElse(s, 0))
      }

      case p @ Product(id, multiplicands) => {
        val operands = multiplicands.map(constructSubAST(_, module, inputParam))
        val rhs = operands.tail.fold(operands.head)(module.mul)
        val variable = module.createVariable(rhs.getType, id)
        module.insertStatement(module.declareVariable(variable, rhs))
        module.readVariable(variable).addAnnotation("spn.graph.depth", graphDepth.getOrElse(p, 0))
      }
    })

  private def double2LNS(value : Double, module : ASTModule) : ASTValue = {
    val falseVal = module.constantValue(BooleanType, false)
    val trueVal = module.constantValue(BooleanType, true)
    if (value == 0.0)
      module.initStruct(LNSType, module.constantValue(PreciseIntegerType, "0x0"), falseVal, trueVal)
    else
      module.initStruct(LNSType,
        module.constantValue(PreciseIntegerType,
          s"0x${LNS(value, config.lnsIntegerBits, config.lnsFractionBits).exp2BigInt.toString(16)}"),
        trueVal, falseVal)
  }

  private def double2LNSSW(value : Double, module : ASTModule) : ASTValue = {
    val LNSSWType = config.lnsSoftwareType match {
      case FLOAT => LNSSWSimTypeFloat
      case _ => LNSSWSimTypeDouble
      // Possibility to add error handling, set double as default for now
    }
    val falseVal = module.constantValue(BooleanType, false)
    val trueVal = module.constantValue(BooleanType, true)
    if (value == 0.0)
      module.initStruct(LNSSWType, module.constantValue(RealType, value), trueVal, trueVal)
    else
      module.initStruct(LNSSWType, module.constantValue(RealType, value), falseVal, trueVal)
  }

  private def double2Posit(value : Double, module : ASTModule) : ASTValue = {
    module.initStruct(PositType, module.constantValue(RealType, value))
  }

  private def double2FPSim(value : Double, module : ASTModule) : ASTValue = {
    module.initStruct(FPSimType, module.constantValue(RealType, value))
  }

  private def double2Fixed(value : Double, module : ASTModule) : ASTValue = {
    module.initStruct(FixedType, module.constantValue(RealType, value))
  }

  private def lns2Double(lns : ASTValue, module : ASTModule) : ASTValue =
    module.call(LNS2Double, lns)

  private val graphDepth : mutable.Map[IRNode, Int] = mutable.Map()

  private def calculateGraphDepth(node : IRNode) : Int = node match {
    case h : Histogram => {
      graphDepth += h -> 0
      0
    }
    case WeightedAddend(a, _) => calculateGraphDepth(a)
    case ws @ WeightedSum(_, addends) => {
      val depth = addends.map(calculateGraphDepth).fold(0){case (a, b) => if(a>b) a else b}+1
      graphDepth += ws -> depth
      depth
    }
    case s @ Sum(_, addends) => {
      val depth = addends.map(calculateGraphDepth).fold(0){case (a, b) => if(a>b) a else b}+1
      graphDepth += s -> depth
      depth
    }
    case p @ Product(_, multiplicands) => {
      val depth = multiplicands.map(calculateGraphDepth).fold(0){case (a, b) => if(a>b) a else b}+1
      graphDepth += p -> depth
      depth
    }
  }

}
