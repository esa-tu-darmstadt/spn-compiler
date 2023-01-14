#include "pipeline/PipelineStep.h"
#include "mlir/IR/BuiltinOps.h"


namespace spnc {

/**
 * This class essentially just wraps exportVerilog() but writes it to 
 * a string instead of a file to allow further processing (for example
 * packing it into a IpXACT file).
*/
class EmitVerilogCode : public StepSingleInput<EmitVerilogCode, mlir::ModuleOp>,
                        public StepWithResult<std::string> {
public:
  using StepSingleInput<EmitVerilogCode, mlir::ModuleOp>::StepSingleInput;

  ExecutionResult executeStep(mlir::ModuleOp *root);

  std::string *result() override { return verilogCode.get(); }

  STEP_NAME("emit-verilog-code")
private:
  std::unique_ptr<std::string> verilogCode = std::make_unique<std::string>();
};

}