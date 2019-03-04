package spn_compiler.backend.software.ast.construct.util

import scala.collection.mutable

class UniqueNameCreator {

  private var existingNames : mutable.Map[String, Int] = mutable.Map()

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
