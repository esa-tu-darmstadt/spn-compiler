#include "CreateVerilogFiles.hpp"

#include "circt/Conversion/ExportVerilog.h"


namespace spnc {

ExecutionResult CreateVerilogFiles::executeStep(mlir::ModuleOp *root) {
  namespace fs = std::filesystem;

  fs::create_directory(config.targetDir);
  fs::create_directory(config.targetDir / config.tmpdirName);
  fs::create_directory(config.targetDir / "src");
  fs::path srcPath = config.targetDir / "src";

  spdlog::info("Exporting verilog sources to: {}", srcPath.string());
  if (failed(::circt::exportSplitVerilog(*root, srcPath.string())))
    return failure("exportSplitVerilog() failed");

  fs::remove(srcPath / "filelist.f");

  // rename top.sv to top.v
  //fs::rename(srcPath / "top.sv", srcPath / "top.v");

  return success();
}

}