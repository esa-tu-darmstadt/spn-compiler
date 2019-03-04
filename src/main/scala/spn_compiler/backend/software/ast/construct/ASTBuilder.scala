package spn_compiler.backend.software.ast.construct

import spn_compiler.backend.software.ast.construct.util.UniqueNameCreator
import spn_compiler.backend.software.ast.nodes.reference.{ASTElementReference, ASTIndexReference, ASTReference, ASTVariableReference}
import spn_compiler.backend.software.ast.nodes.statement.variable.{ASTVariableAssignment, ASTVariableDeclaration}
import spn_compiler.backend.software.ast.nodes.types.{ASTType, ScalarType}
import spn_compiler.backend.software.ast.nodes.value.ASTValue
import spn_compiler.backend.software.ast.nodes.value.access.ASTVariableRead
import spn_compiler.backend.software.ast.nodes.value.constant.ASTConstant
import spn_compiler.backend.software.ast.nodes.value.expression._
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

  //
  // Constants and literals.
  //

  def constantValue[ConstantType <: ScalarType, BaseType <: ConstantType#BaseType]
    (ty : ConstantType, value : BaseType) : ASTConstant[ConstantType, BaseType] =
      new ASTConstant[ConstantType, BaseType](ty, value)

  //
  // Binary and unary expressions.
  //
  def add(leftOp : ASTValue, rightOp : ASTValue) : ASTAddition = new ASTAddition(leftOp, rightOp)

  def sub(leftOp : ASTValue, rightOp : ASTValue) : ASTSubtraction = new ASTSubtraction(leftOp, rightOp)

  def mul(leftOp : ASTValue, rightOp : ASTValue) : ASTMultiplication = new ASTMultiplication(leftOp, rightOp)

  def div(leftOp : ASTValue, rightOp : ASTValue) : ASTDivision = new ASTDivision(leftOp, rightOp)

  def rem(leftOp : ASTValue, rightOp : ASTValue) : ASTRemainder = new ASTRemainder(leftOp, rightOp)

  def and(leftOp : ASTValue, rightOp : ASTValue) : ASTAnd = new ASTAnd(leftOp, rightOp)

  def or(leftOp : ASTValue, rightOp : ASTValue) : ASTOr = new ASTOr(leftOp, rightOp)

  def xor(leftOp : ASTValue, rightOp : ASTValue) : ASTXor = new ASTXor(leftOp, rightOp)

  def cmpEQ(leftOp : ASTValue, rightOp : ASTValue) : ASTCmpEQ = new ASTCmpEQ(leftOp, rightOp)

  def cmpNE(leftOp : ASTValue, rightOp : ASTValue) : ASTCmpNE = new ASTCmpNE(leftOp, rightOp)

  def cmpLT(leftOp : ASTValue, rightOp : ASTValue) : ASTCmpLT = new ASTCmpLT(leftOp, rightOp)

  def cmpLE(leftOp : ASTValue, rightOp : ASTValue) : ASTCmpLE = new ASTCmpLE(leftOp, rightOp)

  def cmpGT(leftOp : ASTValue, rightOp : ASTValue) : ASTCmpGT = new ASTCmpGT(leftOp, rightOp)

  def cmpGE(leftOp : ASTValue, rightOp : ASTValue) : ASTCmpGE = new ASTCmpGE(leftOp, rightOp)

  def neg(op : ASTValue) : ASTNeg = new ASTNeg(op)

  def not(op : ASTValue) : ASTNot = new ASTNot(op)

}
