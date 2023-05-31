#pragma once

#include "Kernel.h"

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

class EmbedAXIStream : public StepSingleInput<EmbedAXIStream, mlir::ModuleOp>,
                       public StepWithResult<mlir::ModuleOp> {
  void setParameters(uint32_t bodyDelay);
public:
  explicit EmbedAXIStream(StepWithResult<mlir::ModuleOp>& circuit):
    StepSingleInput<EmbedAXIStream, mlir::ModuleOp>(circuit) {}

  ExecutionResult executeStep(mlir::ModuleOp *circuit);

  mlir::ModuleOp *result() override { return topModule.get(); }

  STEP_NAME("embed-axistream");
private:
  std::unique_ptr<mlir::ModuleOp> topModule;

  //static ExecutionResult convertFirrtlToHw(mlir::ModuleOp op, circt::hw::HWModuleOp spnBody);
  //void insertCosimTopLevel(mlir::ModuleOp root, uint32_t spnVarCount, uint32_t bitsPerVar, uint32_t resultBitWidth);
};

class ReadyValidWrapper : public Module<ReadyValidWrapper> {
  circt::firrtl::FModuleOp spnBody;
  uint32_t spnVarCount, bitsPerVar, resultWidth, fifoDepth, bodyDelay;
public:
  ReadyValidWrapper(circt::firrtl::FModuleOp spnBody,
    uint32_t spnVarCount, uint32_t bitsPerVar, uint32_t resultWidth, uint32_t fifoDepth, uint32_t bodyDelay):
    Module<ReadyValidWrapper>(
      "ReadyValidWrapper",
      {
        Port("enq", true, readyValidType(withLast(vectorType(uintType(bitsPerVar), spnVarCount)))),
        Port("deq", false, readyValidType(withLast(uintType(resultWidth))))
      },
      spnVarCount, bitsPerVar, resultWidth, fifoDepth, bodyDelay
    ),
    spnBody(spnBody),
    spnVarCount(spnVarCount), bitsPerVar(bitsPerVar), resultWidth(resultWidth), fifoDepth(fifoDepth), bodyDelay(bodyDelay)
    {build();}

  void body();
};

class AXIStreamWrapper : public Module<AXIStreamWrapper> {
  circt::firrtl::FModuleOp spnBody;
  uint32_t spnVarCount, bitsPerVar, resultWidth, fifoDepth, bodyDelay;
  AXIStreamConfig slaveConfig;
  AXIStreamConfig masterConfig;
public:
  AXIStreamWrapper(circt::firrtl::FModuleOp spnBody,
    const AXIStreamConfig& slaveConfig, const AXIStreamConfig& masterConfig,
    uint32_t spnVarCount, uint32_t bitsPerVar, uint32_t resultWidth, uint32_t fifoDepth, uint32_t bodyDelay):
    Module<AXIStreamWrapper>(
      "AXIStreamWrapper",
      {
        Port("AXIS_SLAVE", true, AXIStreamBundleType(slaveConfig)),
        Port("AXIS_MASTER", false, AXIStreamBundleType(masterConfig))
      },
      slaveConfig.dataBits, slaveConfig.userBits, slaveConfig.destBits, slaveConfig.idBits,
      masterConfig.dataBits, masterConfig.userBits, masterConfig.destBits, masterConfig.idBits,
      spnVarCount, bitsPerVar, resultWidth, fifoDepth, bodyDelay
    ),
    spnBody(spnBody),
    slaveConfig(slaveConfig), masterConfig(masterConfig),
    spnVarCount(spnVarCount), bitsPerVar(bitsPerVar), resultWidth(resultWidth), fifoDepth(fifoDepth), bodyDelay(bodyDelay)
    {build();}

  void body();
};

}