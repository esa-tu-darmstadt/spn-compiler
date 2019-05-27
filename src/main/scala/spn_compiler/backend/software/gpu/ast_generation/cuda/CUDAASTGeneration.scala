package spn_compiler.backend.software.gpu.ast_generation.cuda

import spn_compiler.backend.software.ast.construct._
import spn_compiler.backend.software.ast.extensions.cuda.CUDAModule
import spn_compiler.backend.software.ast.extensions.cuda.function.CUDAFunction
import spn_compiler.backend.software.ast.extensions.cuda.predef._
import spn_compiler.backend.software.ast.nodes.function.{ASTFunction, ASTFunctionParameter}
import spn_compiler.backend.software.ast.nodes.module.ASTModule
import spn_compiler.backend.software.ast.nodes.reference.ASTReference
import spn_compiler.backend.software.ast.nodes.statement.ASTStatement
import spn_compiler.backend.software.ast.nodes.types._
import spn_compiler.backend.software.ast.nodes.value.ASTValue
import spn_compiler.backend.software.ast.nodes.variable.ASTVariable
import spn_compiler.backend.software.ast.predef.Ceil
import spn_compiler.graph_ir.nodes._

import scala.collection.mutable

class CUDAASTGeneration {

  def createAST(graph : IRGraph) : CUDAModule = {
    val module = new CUDAModule("spn")
    val inputStructType = module.createStructType("activation", graph.inputVariables.map(v => (v.id, IntegerType)):_*)
    val spnCalc = createSPNDeviceFunction(graph.rootNode, module, inputStructType)
    val spnKernel = createSPNKernel(module, spnCalc, inputStructType)
    val toplevelFunction = createCPUTopLevelFunction(spnKernel, module, inputStructType)
    module
  }

  protected def createCPUTopLevelFunction(spnKernel : CUDAFunction, module : CUDAModule, inputStructType : StructType) : ASTFunction = {
    val numElements = module.createFunctionParameter("num_examples", IntegerType)
    val inputData = module.createFunctionParameter("input_data", module.createArrayType(inputStructType))
    val outputData = module.createFunctionParameter("output_data", module.createArrayType(RealType))
    val topLevelFunction = module.defineLocalFunction("spn_toplevel", VoidType, numElements, inputData, outputData)
    module.setInsertionPoint(topLevelFunction.body)
    val deviceInput = module.createVariable(ArrayType(inputStructType), "deviceInput")
    module.insertStatement(module.declareVariable(deviceInput))
    val deviceOutput = module.createVariable(ArrayType(RealType), "deviceOutput")
    module.insertStatement(module.declareVariable(deviceOutput))
    val inputSize = module.mul(module.sizeOf(inputData), numElements)
    val outputSize = module.mul(module.sizeOf(outputData), numElements)
    module.insertStatement(module.createCallStatement(CUDAMalloc, convert2MallocPointer(module, deviceInput), inputSize))
    module.insertStatement(module.createCallStatement(CUDAMalloc, convert2MallocPointer(module, deviceOutput), outputSize))
    module.insertStatement(module.createCallStatement(CUDAMemCpy, module.convert(deviceInput, ArrayType(VoidType)),
      module.convert(inputData, ArrayType(VoidType)), CUDAMemCpyHostToDevice))
    val gridDim = module.createVariable(CUDADim3Type, "gridDim")
    module.insertStatement(module.dim3(gridDim,
      module.convert(module.call(Ceil,
        module.div(module.convert(numElements, RealType), module.constantValue(RealType, 128.0))), IntegerType)))
    val blockDim = module.createVariable(CUDADim3Type, "blockDim")
    module.insertStatement(module.dim3(blockDim, module.constantValue(IntegerType, 128)))
    module.insertStatement(module.invokeKernel(gridDim, blockDim, spnKernel, numElements, deviceInput, deviceOutput))
    module.insertStatement(module.createCallStatement(CUDAMemCpy, module.convert(outputData, ArrayType(VoidType)),
      module.convert(deviceOutput, ArrayType(VoidType)), outputSize, CUDAMemCpyDeviceToHost))
    topLevelFunction
  }

  private def convert2MallocPointer(module : ASTModule, ref : ASTReference) : ASTValue =
    module.convert(module.addressOf(ref), ArrayType(ArrayType(VoidType)))

  protected def createSPNKernel(module : CUDAModule, spnFunction : CUDAFunction,
                                inputStructType : StructType) : CUDAFunction = {
    val numElements = module.createFunctionParameter("num_examples", IntegerType)
    val inputData = module.createFunctionParameter("gpu_input_data", module.createArrayType(inputStructType))
    val outputData = module.createFunctionParameter("gpu_output_data", module.createArrayType(RealType))
    val spnKernel = module.defineLocalCUDAFunction(CUDAFunction.Global,
      "spn_kernel", VoidType, numElements, inputData, outputData)
    module.setInsertionPoint(spnKernel.body)
    val globalID = module.createVariable(IntegerType, "globalID")
    val calcID = module.add(
      module.mul(module.readElement(CUDABlockDim, "x"), module.readElement(CUDABlockID, "x")),
      module.readElement(CUDAThreadID, "x"))
    module.insertStatement(module.declareVariable(globalID, calcID))
    val ifStmt = module.createIf(module.cmpLT(globalID, numElements))
    module.insertStatement(ifStmt)
    module.setInsertionPoint(ifStmt.thenBranch)
    module.insertStatement(module.assignIndex(outputData, globalID,
      module.call(spnFunction, module.readIndex(inputData, globalID))))
    spnKernel
  }

  protected def createSPNDeviceFunction(spnRoot : IRNode, module : CUDAModule, inputStructType : StructType) : CUDAFunction = {
    val inputParam = module.createFunctionParameter("activation", inputStructType)
    val spnFunction = module.defineLocalCUDAFunction(CUDAFunction.Device,"spn_calc", RealType, inputParam)
    module.setInsertionPoint(spnFunction.body)
    module.insertStatement(module.ret(constructSubAST(spnRoot, module, inputParam)))
    spnFunction
  }

  private val constructedSubGraphs : mutable.Map[IRNode, ASTValue] = mutable.Map()

  protected def constructSubAST(subTreeRoot : IRNode, module : CUDAModule, inputParam : ASTFunctionParameter) : ASTValue =
    constructedSubGraphs.getOrElseUpdate(subTreeRoot, subTreeRoot match {
      case InputVar(id, _) => module.readElement(module.referenceVariable(inputParam), id)

      case Histogram(id, indexVar, buckets) => {
        val activation = constructSubAST(indexVar, module, inputParam)
        val histVar = module.createVariable(RealType, id)
        val decl = module.declareVariable(histVar, module.constantValue(RealType, 0.0))
        module.insertStatement(decl)
        val ifStmt = constructHistogram(module, buckets, activation, histVar)
        module.setInsertionPointAfter(decl)
        module.insertStatement(ifStmt)
        module.readVariable(histVar)
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

  protected def constructHistogram(module : CUDAModule, buckets : List[HistogramBucket],
                                   index : ASTValue, variable : ASTVariable) : ASTStatement = buckets match {
    case Nil => module.assignVariable(variable, module.constantValue(RealType, 0.0))
    case head :: tail => {
      val ifStmt = module.createIf(module.cmpLT(index, module.constantValue(IntegerType, head.upperBound)))
      module.setInsertionPoint(ifStmt.thenBranch)
      module.insertStatement(module.assignVariable(variable, module.constantValue(RealType, head.value)))
      module.setInsertionPoint(ifStmt.elseBranch)
      module.insertStatement(constructHistogram(module, tail, index, variable))
      ifStmt
    }
  }

}
