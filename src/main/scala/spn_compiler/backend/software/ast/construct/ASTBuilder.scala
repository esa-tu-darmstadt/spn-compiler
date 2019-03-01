package spn_compiler.backend.software.ast.construct

import spn_compiler.backend.software.ast.nodes.statement.ASTVariableDeclaration
import spn_compiler.backend.software.ast.nodes.types.{ASTType, ScalarType}
import spn_compiler.backend.software.ast.nodes.value.ASTValue
import spn_compiler.backend.software.ast.nodes.value.constant.ASTConstant
import spn_compiler.backend.software.ast.nodes.value.expression.ASTAddition
import spn_compiler.backend.software.ast.nodes.variable.ASTVariable

import scala.collection.mutable

class ASTBuilder {

  type ASTBuildingException = RuntimeException

  private val variableNames : mutable.Map[String, Int] = mutable.Map[String, Int]()

  private var variables : Set[ASTVariable] = Set.empty

  private def makeUnique(varName : String) : String = {
    if(!variableNames.contains(varName)){
      // If this is the first time we encounter this variable name, we can use the name directly.
      // The next encounter of this name will be suffixed with an index, starting with "100".
      variableNames += varName -> 100
      varName
    }
    else{
      var currentName = varName
      // Iterate and add suffixes to the name until it is unique.
      while(variableNames.contains(currentName)){
        val index = variableNames(currentName)
        // Increment the index for the next encounter of this variable.
        variableNames.update(currentName, index+1)
        currentName = "%s_%d".format(currentName, index)
      }
      currentName
    }
  }

  def createVariable[BaseType, VarType <: ASTType](ty : VarType, name : String) : ASTVariable = {
    val variable = new ASTVariable(ty, makeUnique(name))
    variables = variables + variable
    variable
  }

  def declareVariable[BaseType, VarType <: ASTType](variable : ASTVariable) : ASTVariableDeclaration = {
    if(!variables.contains(variable)){
      throw new ASTBuildingException("Can only declare variable created with this builder before!")
    }
    new ASTVariableDeclaration(variable)
  }

  def constantValue[ConstantType <: ScalarType, BaseType <: ConstantType#BaseType]
    (ty : ConstantType, value : BaseType) : ASTConstant[ConstantType, BaseType] =
      new ASTConstant[ConstantType, BaseType](ty, value)

  def add(leftOp : ASTValue, rightOp : ASTValue): ASTAddition = new ASTAddition(leftOp, rightOp)


}
