package spn_compiler.backend.software.ast.construct

import spn_compiler.backend.software.ast.nodes.statement.ASTVariableDeclaration
import spn_compiler.backend.software.ast.nodes.types.{ASTType, NumericType, ScalarType}
import spn_compiler.backend.software.ast.nodes.value.ASTValue
import spn_compiler.backend.software.ast.nodes.value.constant.ASTConstant
import spn_compiler.backend.software.ast.nodes.value.expression.ASTAddition
import spn_compiler.backend.software.ast.nodes.variable.ASTVariable

import scala.collection.mutable

class ASTBuilder {

  type ASTBuildingException = RuntimeException

  private val variableNames : mutable.Map[String, Int] = mutable.Map[String, Int]()

  private var variables : Set[ASTVariable[_, _]] = Set.empty

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

  def createVariable[BaseType, VarType <: ASTType[BaseType]](ty : VarType, name : String)
    : ASTVariable[BaseType, VarType] = {
    val variable = new ASTVariable[BaseType, VarType](ty, makeUnique(name))
    variables = variables + variable
    variable
  }

  def declareVariable[BaseType, VarType <: ASTType[BaseType]](variable : ASTVariable[BaseType, VarType]) :
    ASTVariableDeclaration[BaseType, VarType] = {
    if(!variables.contains(variable)){
      throw new ASTBuildingException("Can only declare variable created with this builder before!")
    }
    new ASTVariableDeclaration[BaseType, VarType](variable)
  }

  def constantValue[BaseType, ConstantType <: ScalarType[BaseType]]
    (ty : ConstantType, value : BaseType) : ASTConstant[BaseType, ConstantType] =
      new ASTConstant[BaseType, ConstantType](ty, value)

  def add[BaseType, ValueType <: NumericType[BaseType]]
    (leftOp : ASTValue[BaseType, ValueType], rightOp : ASTValue[BaseType, ValueType])
    : ASTAddition[BaseType, ValueType] = new ASTAddition[BaseType, ValueType](leftOp, rightOp)


}
