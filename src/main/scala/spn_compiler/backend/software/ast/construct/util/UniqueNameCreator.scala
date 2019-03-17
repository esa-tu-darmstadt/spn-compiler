package spn_compiler.backend.software.ast.construct.util

import scala.collection.mutable

/**
  * Utility class to compute unique names for entities.
  */
class UniqueNameCreator {

  private var existingNames : mutable.Map[String, Int] = mutable.Map()

  /**
    * Create a unique name, using the given name as base. The method will incrementally add suffixes to the name
    * until the name is unique.
    * @param name Base name.
    * @return Unique name, based on the given name.
    */
  def makeUniqueName(name : String) : String = {
    if(!existingNames.contains(name)){
      // If this is the first time we encounter this variable name, we can use the name directly.
      // The next encounter of this name will be suffixed with an index, starting with "100".
      existingNames += name -> 100
      name
    }
    else{
      val index = existingNames(name)
      existingNames.update(name, index+1)
      makeUniqueName("%s_%d".format(name, index))
    }
  }

}
