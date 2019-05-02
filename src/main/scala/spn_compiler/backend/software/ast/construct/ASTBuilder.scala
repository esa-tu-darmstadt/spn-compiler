package spn_compiler.backend.software.ast.construct

import spn_compiler.backend.software.ast.construct.util.UniqueNameCreator
import spn_compiler.backend.software.ast.nodes.function.{ASTFunction, ASTFunctionParameter, ASTFunctionPrototype}
import spn_compiler.backend.software.ast.nodes.reference._
import spn_compiler.backend.software.ast.nodes.statement.control_flow.{ASTCallStatement, ASTForLoop, ASTIfStatement, ASTReturnStatement}
import spn_compiler.backend.software.ast.nodes.statement.variable.{ASTVariableAssignment, ASTVariableDeclaration}
import spn_compiler.backend.software.ast.nodes.statement.{ASTBlockStatement, ASTStatement}
import spn_compiler.backend.software.ast.nodes.types.{ASTType, ScalarType, StructType}
import spn_compiler.backend.software.ast.nodes.value.ASTValue
import spn_compiler.backend.software.ast.nodes.value.access.ASTVariableRead
import spn_compiler.backend.software.ast.nodes.value.constant.{ASTArrayInit, ASTConstant, ASTStructInit}
import spn_compiler.backend.software.ast.nodes.value.expression._
import spn_compiler.backend.software.ast.nodes.value.function.ASTCallExpression
import spn_compiler.backend.software.ast.nodes.value.type_conversion.ASTTypeConversion
import spn_compiler.backend.software.ast.nodes.variable.ASTVariable

import scala.collection.mutable.ListBuffer

/**
  * Interface for creating AST nodes. The constructors of the AST nodes are private, so
  * the methods of the [[ASTBuilder]] are the only interface to create new AST nodes.
  * This allows us to maintain important relationships in this builder.
  *
  * Constructed [[ASTStatement]]s will be inserted automatically at a given insertion point. The insertion point can
  * be moved around using the specified methods.
  */
trait ASTBuilder {

  type ASTBuildingException = RuntimeException

  //
  // Function handling
  //
  protected val localFunctions : ListBuffer[ASTFunction] = ListBuffer()

  /**
    * Create a new, externally defined function.
    * @param name Name of the function.
    * @param returnType Return type of the function.
    * @param parameters Parameter types of the function (unnamed parameters).
    * @return [[ASTFunctionPrototype]] as signature of the externally defined function.
    */
  def declareExternalFunction(name : String, returnType : ASTType, parameters : ASTType*) : ASTFunctionPrototype =
    new ASTFunctionPrototype(name, returnType, parameters:_*)

  /**
    * Create a new formal parameter.
    * @param name Name of the parameter.
    * @param ty [[ASTType]] of the parameter.
    * @return [[ASTFunctionParameter]] with given name and type.
    */
  def createFunctionParameter(name : String, ty : ASTType) : ASTFunctionParameter = new ASTFunctionParameter(name, ty)

  /**
    * Define a local (i.e. defined in this module) function.
    * @param name Name of the function.
    * @param returnType Return type of the function.
    * @param parameters [[ASTFunctionParameter]]s as formal parameters of the function.
    * @return [[ASTFunction]] with given name, return type and formal parameters.
    */
  def defineLocalFunction(name : String, returnType : ASTType, parameters : ASTFunctionParameter*) : ASTFunction = {
    val func = new ASTFunction(name, returnType, parameters:_*)
    localFunctions += func
    func
  }


  //
  // Maintain insertion point
  //
  protected case class ASTInsertionPoint(block : Option[ASTBlockStatement], stmt : Option[ASTStatement])

  protected var insertionPoint : ASTInsertionPoint = ASTInsertionPoint(None, None)

  /**
    * Set the insertion point for this builder to the end of the given block.
    * @param block [[ASTBlockStatement]], next statement(s) will be appended to its end.
    */
  def setInsertionPoint(block : ASTBlockStatement) : Unit = {
    insertionPoint = ASTInsertionPoint(Some(block), None)
  }

  /**
    * Set the insertion point for this builder right in front of the given statement.
    * @param stmt [[ASTStatement]], next statement(s) will be inserted in front of this statement.
    */
  def setInsertionPointBefore(stmt : ASTStatement) : Unit = {
    require(stmt.block.isDefined, "No block defined for statement!")
    insertionPoint = ASTInsertionPoint(stmt.block, Some(stmt))
  }

  /**
    * Set the insertion point for this builder right after the given statement.
    * @param stmt [[ASTStatement]], next statement(s) will be inserted right after this statement.
    */
  def setInsertionPointAfter(stmt : ASTStatement) : Unit = {
    require(stmt.block.isDefined, "No block defined for statement!")
    insertionPoint = ASTInsertionPoint(stmt.block, stmt.block.get.getNextStatement(stmt))
  }

  /**
    * Delete the given statement. If this statement was used as insertion point, the insertion point will
    * be set to the end of the block.
    * @param stmt Statement to delete.
    */
  def deleteStatement(stmt : ASTStatement) : Unit = {
    if(stmt.block.isDefined){
      stmt.block.get.delete(stmt)
      if(insertionPoint.stmt.contains(stmt)){
        setInsertionPoint(stmt.block.get)
      }
    }
  }

  def insertStatement[Stmt <: ASTStatement](stmt : Stmt) : Stmt =
    (insertionPoint.block, insertionPoint.stmt) match {
      case (Some(block), Some(insertBefore)) => block.insertBefore(insertBefore, stmt)
      case (Some(block), None) => block.append(stmt)
      case _ => stmt
  }

  //
  // Control flow handling
  //

  /**
    * Create an if-statement with the given test expression.
    * @param testExpression Boolean test expression.
    * @return New [[ASTIfStatement]]
    */
  def createIf(testExpression : ASTValue): ASTIfStatement = new ASTIfStatement(testExpression)

  /**
    * Create a for-loop with the following header: for(IVar = IVal; TVal; IncrVar = IncrVal).
    * @param initVar IVar.
    * @param initValue IVal.
    * @param testValue Boolean TVal;
    * @param incrVar IncrVar;
    * @param incrValue IncrVal;
    * @return New [[ASTForLoop]]
    */
  def forLoop(initVar : Option[ASTReference], initValue : Option[ASTValue], testValue : ASTValue,
                 incrVar : Option[ASTReference], incrValue : Option[ASTValue]) : ASTForLoop =
    new ASTForLoop(initVar, initValue, testValue, incrVar, incrValue)

  /**
    * Create a for-loop with the following header: for(Var = LB; Var < UB; Var = Var + Stride).
    * @param variable   Var
    * @param lowerBound LB
    * @param upperBound UB
    * @param stride     Stride
    * @return New [[ASTForLoop]]
    */
  def forLoop(variable : ASTVariable, lowerBound : ASTValue, upperBound : ASTValue, stride : ASTValue) : ASTForLoop = {
    val ref = referenceVariable(variable)
    val comparison = cmpLT(readVariable(ref), upperBound)
    val increment = add(readVariable(ref), stride)
    new ASTForLoop(Some(ref), Some(lowerBound), comparison, Some(ref), Some(increment))
  }

  /**
    * Create a call '''statement''' from a call '''expression''', discarding the return value if necessary.
    * @param call [[ASTCallExpression]] for the actual call.
    * @return New [[ASTCallStatement]]
    */
  def createCallStatement(call : ASTCallExpression) : ASTCallStatement = new ASTCallStatement(call)

  /**
    * Create a call '''statement''' for the given function with the given parameters, discarding the return value
    * if necessary.
    * @param function [[ASTFunctionPrototype]] to call.
    * @param parameters Actual parameter values.
    * @return New [[ASTCallStatement]].
    */
  def createCallStatement(function : ASTFunctionPrototype, parameters : ASTValue*) : ASTCallStatement =
    new ASTCallStatement(new ASTCallExpression(function, parameters:_*))

  /**
    * Create a call '''expression''', calling the given function with the given parameters.
    * @param function [[ASTFunctionPrototype]] to call.
    * @param parameters Actual parameter values.
    * @return New [[ASTCallExpression]].
    */
  def call(function : ASTFunctionPrototype, parameters : ASTValue*) : ASTCallExpression =
    new ASTCallExpression(function, parameters:_*)

  /**
    * Create a return statement, returning the given value.
    * @param returnValue Return value.
    * @return New [[ASTReturnStatement]].
    */
  def ret(returnValue : ASTValue) : ASTReturnStatement = new ASTReturnStatement(returnValue)


  //
  // Variable handling
  //

  private val variableNameCreator = new UniqueNameCreator

  private var variables : Set[ASTVariable] = Set.empty

  /**
    * Create a new variable with the given type, using the given name as base
    * for a unique name.
    * @param ty [[ASTType]] of the variable.
    * @param name Base name of the variable. The name will be made unique by appending suffixes if necessary.
    * @return [[ASTVariable]].
    */
  def createVariable(ty : ASTType, name : String) : ASTVariable = {
    val variable = new ASTVariable(ty, variableNameCreator.makeUniqueName(name))
    variables = variables + variable
    variable
  }

  /**
    * Declare a variable.
    * @param variable [[ASTVariable]] to declare. Must have been constructed with this builder before.
    * @return [[ASTVariableDeclaration]].
    */
  def declareVariable(variable : ASTVariable) : ASTVariableDeclaration = {
    if(!variables.contains(variable)){
      throw new ASTBuildingException("Can only declare variable created with this builder before!")
    }
    new ASTVariableDeclaration(variable)
  }

  /**
    * Declare a variable and initialize it to the given value.
    * @param variable [[ASTVariable]] to declare. Must have been constructed with this builder before.
    * @param initValue Initial value of the declared variable.
    * @return [[ASTVariableDeclaration]].
    */
  def declareVariable(variable : ASTVariable, initValue : ASTValue) : ASTVariableDeclaration = {
    if(!variables.contains(variable)){
      throw new ASTBuildingException("Can only declare variable created with this builder before!")
    }
    new ASTVariableDeclaration(variable, Some(initValue))
  }

  protected val globalVariables : ListBuffer[ASTVariableDeclaration] = ListBuffer()

  /**
    * Declare the given variable as global variable in this module.
    * @param variable [[ASTVariable]] to declare as global variable.
    */
  def declareGlobalVariable(variable : ASTVariable) : Unit = {
    if(!variables.contains(variable)){
      throw new ASTBuildingException("Can only declare variable created with this builder before!")
    }
    val declaration = new ASTVariableDeclaration(variable)
    globalVariables += declaration
  }

  /**
    * Declare the given variable as global variable in this module.
    * @param variable [[ASTVariable]] to declare as global variable.
    * @param initValue Initial value of the variable.
    */
  def declareGlobalVariable(variable : ASTVariable, initValue : ASTValue) : Unit = {
    if(!variables.contains(variable)){
      throw new ASTBuildingException("Can only declare variable created with this builder before!")
    }
    val declaration = new ASTVariableDeclaration(variable, Some(initValue))
    globalVariables += declaration
  }

  /**
    * Reference the given entity.
    * @param variable [[ASTReferencable]] to reference.
    * @return [[ASTReference]].
    */
  def referenceVariable(variable : ASTReferencable) : ASTVariableReference = new ASTVariableReference(variable)

  /**
    * Reference the given index of the given entity.
    * @param reference [[ASTReference]], base entity to reference.
    * @param index Index to reference.
    * @return [[ASTIndexReference]].
    */
  def referenceIndex(reference : ASTReference, index : ASTValue) : ASTIndexReference =
    new ASTIndexReference(reference, index)

  /**
    * Reference the given structure element (by name) of the given entity.
    * @param reference [[ASTReference]], base entity to reference.
    * @param element Name of the element to reference.
    * @return [[ASTElementReference]].
    */
  def referenceElement(reference : ASTReference, element : String) : ASTElementReference =
    new ASTElementReference(reference, element)

  /**
    * Read the value of the given reference to some entity.
    * @param reference [[ASTReference]] to read the value from.
    * @return [[ASTVariableRead]].
    */
  def readVariable(reference : ASTReference) : ASTVariableRead = new ASTVariableRead(reference)

  /**
    * Read the given index from the given reference to some entity.
    * @param reference [[ASTReference]], base entity to read the index from.
    * @param index Index to read from.
    * @return [[ASTVariableRead]].
    */
  def readIndex(reference : ASTReference, index : ASTValue) : ASTVariableRead =
    readVariable(referenceIndex(reference, index))

  /**
    * Read the given element (by name) from the given reference to some entity.
    * @param reference [[ASTReference]], base entity to read the element from.
    * @param element Name of the element to read.
    * @return [[ASTVariableRead]].
    */
  def readElement(reference : ASTReference, element : String) : ASTVariableRead =
    readVariable(referenceElement(reference, element))

  /**
    * Assign the given value to the given reference to some entity.
    * @param reference [[ASTReference]] to assign to.
    * @param value Value to assign.
    * @return [[ASTVariableAssignment]].
    */
  def assignVariable(reference : ASTReference, value : ASTValue) : ASTVariableAssignment =
    new ASTVariableAssignment(reference, value)

  /**
    * Assign the given value to the given index of the given reference to some entity.
    * @param reference [[ASTReference]], base entity to assign the index.
    * @param index Index to assign to.
    * @param value Value to assign.
    * @return [[ASTVariableAssignment]].
    */
  def assignIndex(reference : ASTReference, index : ASTValue, value : ASTValue) : ASTVariableAssignment =
    new ASTVariableAssignment(referenceIndex(reference, index), value)

  /**
    * Assign the given value to the given element (by name) of the given reference to some entity.
    * @param reference [[ASTReference]], base entity to assign the element.
    * @param element Name of the element to assign to.
    * @param value Value to assign.
    * @return [[ASTVariableAssignment]].
    */
  def assignElement(reference : ASTReference, element : String, value : ASTValue) : ASTVariableAssignment =
    new ASTVariableAssignment(referenceElement(reference, element), value)

  //
  // Constants and literals.
  //

  /**
    * Create a new constant literal with the given type and value.
    * @param ty [[ASTType]] of the constant literal.
    * @param value Value of the constant literal (Scala-land).
    * @tparam ConstantType [[ASTType]] of the constant literal.
    * @tparam BaseType Scala-land type of the value.
    * @return [[ASTConstant]], constant literal.
    */
  def constantValue[ConstantType <: ScalarType, BaseType <: ConstantType#BaseType]
    (ty : ConstantType, value : BaseType) : ASTConstant[ConstantType, BaseType] =
      new ASTConstant[ConstantType, BaseType](ty, value)

  /**
    * Create an array initializer ("array-literal") with the given values.
    * @param values Values.
    * @return [[ASTArrayInit]].
    */
  def initArray(values : ASTValue*) : ASTArrayInit = new ASTArrayInit(values:_*)

  /**
    * Create an array initializer ("array-literal") with the given '''Scala-land''' values by creating
    * [[ASTConstant]] for each of the values to form the initializer.
    * @param ty [[ASTType]], element type of the array.
    * @param values Scala-land values.
    * @tparam ConstantType [[ASTType]] as element type of the array.
    * @tparam BaseType Scala-land type of the values.
    * @return [[ASTArrayInit]].
    */
  def initArray[ConstantType <: ScalarType, BaseType <: ConstantType#BaseType](ty : ConstantType, values : BaseType*)
    : ASTArrayInit = new ASTArrayInit(values.map(constantValue(ty, _)):_*)

  /**
    * Create a structure initializer ("struct-literal") with the given values.
    * @param structType [[StructType]], type of the struct.
    * @param values Values.
    * @return [[ASTStructInit]].
    */
  def initStruct(structType: StructType, values : ASTValue*) : ASTStructInit = new ASTStructInit(structType, values:_*)

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

  //
  // Type conversions.
  //
  /**
    * Create a type conversion converting a value of a type to the given type.
    * '''Warning:''' This may include loss of precision or even overflow if the target type cannot hold
    * values with same precision.
    * @param op Value to convert.
    * @param targetType Target type of the conversion.
    * @return [[ASTTypeConversion]].
    */
  def convert(op : ASTValue, targetType : ASTType) : ASTTypeConversion = new ASTTypeConversion(op, targetType)

}
