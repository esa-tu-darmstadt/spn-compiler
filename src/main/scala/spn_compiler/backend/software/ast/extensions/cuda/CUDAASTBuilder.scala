package spn_compiler.backend.software.ast.extensions.cuda

import spn_compiler.backend.software.ast.construct.ASTBuilder
import spn_compiler.backend.software.ast.extensions.cuda.function.CUDAFunction
import spn_compiler.backend.software.ast.extensions.cuda.function.CUDAFunction.CUDAFunctionScope
import spn_compiler.backend.software.ast.extensions.cuda.statement.{CUDADim3Init, CUDAKernelInvocation}
import spn_compiler.backend.software.ast.nodes.function.ASTFunctionParameter
import spn_compiler.backend.software.ast.nodes.types.{ASTType, IntegerType}
import spn_compiler.backend.software.ast.nodes.value.ASTValue
import spn_compiler.backend.software.ast.nodes.value.constant.ASTConstant
import spn_compiler.backend.software.ast.nodes.variable.ASTVariable

trait CUDAASTBuilder extends ASTBuilder {

  def dim3(variable : ASTVariable,
           x : ASTValue = new ASTConstant(IntegerType, 1),
           y : ASTValue = new ASTConstant(IntegerType, 1),
           z : ASTValue = new ASTConstant(IntegerType, 1)) : CUDADim3Init =
    new CUDADim3Init(variable, x, y, z)

  def invokeKernel(gridLayout : ASTValue, blockLayout : ASTValue, kernel : CUDAFunction,
                   params : ASTValue*) : CUDAKernelInvocation =
    new CUDAKernelInvocation(gridLayout, blockLayout, kernel, params:_*)

  /**
    * Define a local (i.e. defined in this module) function.
    * @param name Name of the function.
    * @param returnType Return type of the function.
    * @param parameters [[ASTFunctionParameter]]s as formal parameters of the function.
    * @return [[CUDAFunction]] with given name, return type and formal parameters.
    */
  def defineLocalCUDAFunction(scope : CUDAFunctionScope, name : String, returnType : ASTType,
                              parameters : ASTFunctionParameter*) : CUDAFunction = {
    val func = new CUDAFunction(scope, name, returnType, parameters:_*)
    localFunctions += func
    func
  }
}
