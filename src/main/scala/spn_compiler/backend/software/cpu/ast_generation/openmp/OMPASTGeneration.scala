package spn_compiler.backend.software.cpu.ast_generation.openmp

import spn_compiler.backend.software.ast.construct._
import spn_compiler.backend.software.ast.extensions.openmp.OMPModule
import spn_compiler.backend.software.ast.nodes.function.ASTFunction
import spn_compiler.backend.software.ast.nodes.module.ASTModule
import spn_compiler.backend.software.ast.nodes.types.{IntegerType, RealType, StructType, VoidType}
import spn_compiler.backend.software.cpu.ast_generation.serial.ASTGeneration
import spn_compiler.graph_ir.nodes.IRGraph


class OMPASTGeneration extends ASTGeneration {

  override def createAST(graph : IRGraph) : ASTModule = {
    val module = new OMPModule("spn")
    val inputStructType = module.createStructType("activation",
      graph.inputVariables.map(v => (s"input_${v.id}", IntegerType)):_*)
    val spnFunction = createSPNFunction(graph.rootNode, module, inputStructType)
    val toplevelFunction = createTopLevelFunction(spnFunction, module, inputStructType)
    module
  }

  protected def createTopLevelFunction(spnFunction: ASTFunction,
                                                module: OMPModule, inputStructType: StructType): ASTFunction = {
    val numElements = module.createFunctionParameter("num_examples", IntegerType)
    val inputData = module.createFunctionParameter("input_data", module.createArrayType(inputStructType))
    val outputData = module.createFunctionParameter("output_data", module.createArrayType(RealType))
    val topLevelFunction = module.defineLocalFunction("spn_toplevel", VoidType, numElements, inputData, outputData)
    module.setInsertionPoint(topLevelFunction.body)
    val loopVar = module.createVariable(IntegerType, "i")
    module.insertStatement(module.declareVariable(loopVar))
    val constantZero = module.constantValue(IntegerType, 0)
    val forLoop = module.forLoop(loopVar, constantZero,
      numElements, module.constantValue(IntegerType, 1))
    val ompParrallelFor = module.ompParallelFor(forLoop)
    module.insertStatement(ompParrallelFor)
    module.addClause(ompParrallelFor, module.firstPrivate(inputData, outputData))
    module.setInsertionPoint(forLoop.body)
    module.insertStatement(module.assignIndex(outputData, loopVar,
      module.call(spnFunction, module.readIndex(inputData, loopVar))))
    topLevelFunction
  }
}
