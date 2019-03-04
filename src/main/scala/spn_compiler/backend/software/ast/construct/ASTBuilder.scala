package spn_compiler.backend.software.ast.construct

import spn_compiler.backend.software.ast.construct.util.UniqueNameCreator
import spn_compiler.backend.software.ast.nodes.reference.{ASTElementReference, ASTIndexReference, ASTReference, ASTVariableReference}
import spn_compiler.backend.software.ast.nodes.statement.variable.{ASTVariableAssignment, ASTVariableDeclaration}
import spn_compiler.backend.software.ast.nodes.types.{ASTType, ScalarType}
import spn_compiler.backend.software.ast.nodes.value.ASTValue
import spn_compiler.backend.software.ast.nodes.value.access.ASTVariableRead
import spn_compiler.backend.software.ast.nodes.value.constant.ASTConstant
import spn_compiler.backend.software.ast.nodes.value.expression.ASTAddition
import spn_compiler.backend.software.ast.nodes.variable.ASTVariable

class ASTBuilder {

  type ASTBuildingException = RuntimeException

  //
  // Variable handling
  //

  private val variableNameCreator = new UniqueNameCreator

  private var variables : Set[ASTVariable] = Set.empty


  def createVariable(ty : ASTType, name : String) : ASTVariable = {
    val variable = new ASTVariable(ty, variableNameCreator.makeUniqueName(name))
    variables = variables + variable
    variable
  }

  def declareVariable(variable : ASTVariable) : ASTVariableDeclaration = {
    if(!variables.contains(variable)){
      throw new ASTBuildingException("Can only declare variable created with this builder before!")
    }
    new ASTVariableDeclaration(variable)
  }

  def referenceVariable(variable : ASTVariable) : ASTVariableReference = new ASTVariableReference(variable)

  def referenceIndex(reference : ASTReference, index : ASTValue) : ASTIndexReference =
    new ASTIndexReference(reference, index)

  def referenceElement(reference : ASTReference, element : String) : ASTElementReference =
    new ASTElementReference(reference, element)

  def readVariable(reference : ASTReference) : ASTVariableRead = new ASTVariableRead(reference)

  def readIndex(reference : ASTReference, index : ASTValue) : ASTVariableRead =
    readVariable(referenceIndex(reference, index))

  def readElement(reference : ASTReference, element : String) : ASTVariableRead =
    readVariable(referenceElement(reference, element))

  def assignVariable(reference : ASTReference, value : ASTValue) : ASTVariableAssignment =
    new ASTVariableAssignment(reference, value)

  def assignIndex(reference : ASTReference, index : ASTValue, value : ASTValue) : ASTVariableAssignment =
    new ASTVariableAssignment(referenceIndex(reference, index), value)

  def assignElement(reference : ASTReference, element : String, value : ASTValue) : ASTVariableAssignment =
    new ASTVariableAssignment(referenceElement(reference, element), value)

  def constantValue[ConstantType <: ScalarType, BaseType <: ConstantType#BaseType]
    (ty : ConstantType, value : BaseType) : ASTConstant[ConstantType, BaseType] =
      new ASTConstant[ConstantType, BaseType](ty, value)

  def add(leftOp : ASTValue, rightOp : ASTValue): ASTAddition = new ASTAddition(leftOp, rightOp)

}
