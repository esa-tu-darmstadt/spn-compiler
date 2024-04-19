#include "EmitVerilogCode.h"

#include "circt/Conversion/ExportVerilog.h"
#include "llvm/Support/raw_ostream.h"


namespace spnc {

ExecutionResult EmitVerilogCode::executeStep(mlir::ModuleOp *root) {
  llvm::raw_string_ostream os(*verilogCode);

  root->dump();

  if (mlir::failed(circt::exportVerilog(*root, os)))
    return failure("circt::exportVerilog() failed!");

  return success();
}

}