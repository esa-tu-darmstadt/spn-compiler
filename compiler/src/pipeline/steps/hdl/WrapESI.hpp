#pragma once

#include "pipeline/steps/mlir/MLIRPassPipeline.h"

#include "pipeline/PipelineStep.h"
#include "toolchain/MLIRToolchain.h"
#include "mlir/IR/BuiltinOps.h"

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"

#include "circt/Dialect/FIRRTL/FIRParser.h"

#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESITypes.h"

#include "mlir/Transforms/DialectConversion.h"

#include <firp/AXIStream.hpp>
#include <firp/ShiftRegister.hpp>
#include <firp/firpQueue.hpp>

#include <filesystem>

namespace spnc {

using namespace firp;
using namespace firp::axis;

class WrapESI : public StepSingleInput<WrapESI, mlir::ModuleOp>,
                       public StepWithResult<mlir::ModuleOp> {
  std::string topName;
  // TODO
  bool doWrapEndpoint = true;
public:
  explicit WrapESI(StepWithResult<mlir::ModuleOp>& root, const std::string &topName):
    StepSingleInput<WrapESI, mlir::ModuleOp>(root), topName(topName) {}

  ExecutionResult executeStep(mlir::ModuleOp *root);

  mlir::ModuleOp *result() override { return topModule.get(); }

  STEP_NAME("wrap-esi");
private:
  std::unique_ptr<mlir::ModuleOp> topModule;

  // returns and empty ModuleOp if not modules named topName is found or it is not unique
  circt::hw::HWModuleOp findTop(mlir::ModuleOp root);
  void wrapEndpoint(circt::hw::HWModuleOp esiWrapper, mlir::ModuleOp root);
};

}