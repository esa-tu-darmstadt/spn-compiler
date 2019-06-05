package spn_compiler.backend.software.ast.construct

import spn_compiler.backend.software.ast.construct.util.UniqueNameCreator
import spn_compiler.backend.software.ast.nodes.types.{ASTType, ArrayType, StructType}

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

trait ASTTypeContext {

  private val structNameCreator = new UniqueNameCreator

  protected val structTypes : ListBuffer[StructType] = ListBuffer[StructType]()

  def createStructType(name : String, _elements : (String, ASTType)*): StructType = {
    val structType = new StructType(structNameCreator.makeUniqueName(name), _elements.toList)
    structTypes += structType
    structType
  }

  // TODO: Check if we really need to maintain this map or if we just rely on equals/hashcode.
  protected val arrayTypes : mutable.Map[ASTType, ArrayType] = mutable.Map()

  def createArrayType(elementType : ASTType) : ASTType =
    arrayTypes.getOrElseUpdate(elementType, ArrayType(elementType))

}
