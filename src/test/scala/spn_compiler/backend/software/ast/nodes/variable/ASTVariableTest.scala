package spn_compiler.backend.software.ast.nodes.variable

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}
import spn_compiler.backend.software.ast.nodes.module.ASTModule
import spn_compiler.backend.software.ast.nodes.reference.{ASTElementReference, ASTIndexReference, ASTVariableReference}
import spn_compiler.backend.software.ast.nodes.statement.variable.{ASTVariableAssignment, ASTVariableDeclaration}
import spn_compiler.backend.software.ast.nodes.types.IntegerType
import spn_compiler.backend.software.ast.nodes.value.access.ASTVariableRead

@RunWith(classOf[JUnitRunner])
class ASTVariableTest extends FlatSpec with Matchers {

  val builder = new ASTModule("test-dummy")
  val variable = builder.createVariable(IntegerType, "var")
  val five = builder.constantValue(IntegerType, 5)

  //
  // Basic variables
  //
  "A variable" should "be creatable through the ASTBuilder interface" in {
    "val builder = new ASTModule(\"test-dummy\")\n" +
      "  val variable = builder.createVariable(IntegerType, \"var\")" should compile
  }

  it should "pattern-match" in {
    val ASTVariable(IntegerType, name) = variable
    name should be("var")
  }

  "Variable names" should "be unique" in {
    val variable2 = builder.createVariable(IntegerType, "var")
    val ASTVariable(IntegerType, name) = variable2
    name should not be "var"
    val variable3 = builder.createVariable(IntegerType, "var_101")
    val variable4 = builder.createVariable(IntegerType, "var")
    val ASTVariable(IntegerType, name2) = variable4
    name2 should not be "var_101"
  }

  //
  // Variable declarations
  //
  "A variable-declaration" should "be constructable through the ASTBuilder interface" in {
    "val builder = new ASTModule(\"test-dummy\")\n" +
      "val variable = builder.createVariable(IntegerType, \"var\")\n" +
      "val declaration = builder.declareVariable(variable)" should compile
  }

  it should "pattern-match" in {
    val variable = builder.createVariable(IntegerType, "var")
    val declaration = builder.declareVariable(variable)
    val ASTVariableDeclaration(variable2, None) = declaration
    variable2 should be(variable)
  }

  it should "not be constructable with an unknown variable" in {
    val variable2 = new ASTVariable(IntegerType, "var2")
    an [RuntimeException] should be thrownBy builder.declareVariable(variable2)
  }

  //
  // Variable reading and assigning
  //
  "A scalar reference" should "be constructable from a variable" in {
    val reference = builder.referenceVariable(variable)
    val ASTVariableReference(IntegerType, testVar) = reference
    testVar should be(variable)
  }

  "A scalar read" should "be constructable from a reference" in {
    val reference = builder.referenceVariable(variable)
    val read = builder.readVariable(reference)
    val ASTVariableRead(IntegerType, testRef) = read
    testRef should be(reference)
  }

  "A scalar read" should "be constructable directly from a variable" in {
    // This implicitly tests the implicit conversion from variable to reference.
    import spn_compiler.backend.software.ast.construct._
    val read = builder.readVariable(variable)
    val ASTVariableRead(IntegerType, ASTVariableReference(IntegerType, testVar)) = read
    testVar should be(variable)
  }

  "A scalar assignment" should "be constructable from a variable" in {
    // This implicitly tests the implicit conversion from variable to reference.
    import spn_compiler.backend.software.ast.construct._
    val assign = builder.assignVariable(variable, five)
    val ASTVariableAssignment(ASTVariableReference(IntegerType, testVar), testValue) = assign
    testVar should be(variable)
    testValue should be(five)
  }

  val arrayType = builder.createArrayType(IntegerType)
  val arrayVar = builder.createVariable(arrayType, "arr")
  val index = builder.constantValue(IntegerType, 1)

  "An array reference" should "be constructable from a variable" in {
    // This implicitly tests the implicit conversion from variable to reference.
    import spn_compiler.backend.software.ast.construct._
    val reference = builder.referenceIndex(arrayVar, index)
    val ASTIndexReference(IntegerType, ASTVariableReference(ty, testVar), testIndex) = reference
    ty should be(arrayType)
    testVar should be(arrayVar)
    testIndex should be(index)
  }

  "An index read" should "be constructable from a reference" in {
    val reference = builder.referenceVariable(arrayVar)
    val read = builder.readIndex(reference, index)
    val ASTVariableRead(IntegerType, ASTIndexReference(IntegerType, testRef, testIndex)) = read
    testRef should be(reference)
    testIndex should be(index)
  }

  "An index read" should "be constructable directly from a variable" in {
    // This implicitly tests the implicit conversion from variable to reference.
    import spn_compiler.backend.software.ast.construct._
    val read = builder.readIndex(arrayVar, index)
    val ASTVariableRead(IntegerType, ASTIndexReference(IntegerType, ASTVariableReference(ty, testVar), testIndex)) = read
    ty should be(arrayType)
    testVar should be(arrayVar)
    testIndex should be(index)
  }

  "An index assignment" should "be constructable directly from a variable" in {
    // This implicitly tests the implicit conversion from variable to reference.
    import spn_compiler.backend.software.ast.construct._
    val assign = builder.assignIndex(arrayVar, index, five)
    val ASTVariableAssignment(ASTIndexReference(IntegerType, ASTVariableReference(ty, testVar), testIndex), testValue) = assign
    ty should be(arrayType)
    testVar should be(arrayVar)
    testIndex should be(index)
    testValue should be(five)
  }

  val element = "elem1"
  val structType = builder.createStructType("struct_type", (element, IntegerType))
  val structVar = builder.createVariable(structType, "strct")

  "A struct reference" should "be constructable from a variable" in {
    // This implicitly tests the implicit conversion from variable to reference.
    import spn_compiler.backend.software.ast.construct._
    val reference = builder.referenceElement(structVar, element)
    val ASTElementReference(IntegerType, ASTVariableReference(ty, testVar), testElem) = reference
    ty should be(structType)
    testVar should be(structVar)
    testElem should be(element)
  }

  "An element read" should "be constructable from a reference" in {
    val reference = builder.referenceVariable(structVar)
    val read = builder.readElement(reference, element)
    val ASTVariableRead(IntegerType, ASTElementReference(IntegerType, testRef, testElement)) = read
    testRef should be(reference)
    testElement should be(element)
  }

  "An element read" should "be constructable directly from a variable" in {
    // This implicitly tests the implicit conversion from variable to reference.
    import spn_compiler.backend.software.ast.construct._
    val read = builder.readElement(structVar, element)
    val ASTVariableRead(IntegerType, ASTElementReference(IntegerType, ASTVariableReference(ty, testVar), testElement)) = read
    ty should be(structType)
    testVar should be(structVar)
    testElement should be(element)
  }

  "An element assignment" should "be constructable directly from a variable" in {
    // This implicitly tests the implicit conversion from variable to reference.
    import spn_compiler.backend.software.ast.construct._
    val read = builder.assignElement(structVar, element, five)
    val ASTVariableAssignment(ASTElementReference(IntegerType, ASTVariableReference(ty, testVar), testElement), testValue) = read
    ty should be(structType)
    testVar should be(structVar)
    testElement should be(element)
    testValue should be(five)
  }

}
