package spn_compiler.util.file

import java.io.File

import spn_compiler.util.logging.Logging

object FileUtil extends Logging{

  def getTmpDirectory : File = new File(System.getProperty("java.io.tmpdir")).getAbsoluteFile

  def createScratchpadDirectory : File = {
    val hashCode = Math.abs(System.currentTimeMillis().hashCode())
    val dir = getTmpDirectory.toPath.resolve(hashCode.toString).toFile
    dir.mkdir()
    dir
  }

  def createFileInDirectory(dir : File, fileName : String) : File = {
    if(!dir.exists()){
      error(s"Cannot create file $fileName, $dir does not exist")
    }
    if(!dir.isDirectory){
      error(s"Cannot create file $fileName, $dir is not a directory")
    }
    dir.getAbsoluteFile.toPath.resolve(fileName).toFile
  }

  def createFileInSameDirectory(entity : File, fileName : String) : File =
    entity.toPath.resolveSibling(fileName).toFile

  def getParentDirectory(entity : File) : File = entity.getAbsoluteFile.toPath.getParent.toFile

}
