#pragma once

#include "Kernel.h"

#include "pipeline/steps/mlir/MLIRPassPipeline.h"

#include "pipeline/PipelineStep.h"
#include "toolchain/MLIRToolchain.h"
#include "mlir/IR/BuiltinOps.h"

#include <firp/AXI4.hpp>
#include <firp/AXI4Lite.hpp>
#include <firp/AXIStream.hpp>


namespace spnc {

class CreateAXIStreamMapper : public StepSingleInput<CreateAXIStreamMapper, mlir::ModuleOp>,
                              public StepWithResult<Kernel> {
  //
public:
  explicit CreateAXIStreamMapper(StepWithResult<mlir::ModuleOp>& root):
    StepSingleInput<CreateAXIStreamMapper, mlir::ModuleOp>(root) {}

  ExecutionResult executeStep(mlir::ModuleOp *root);

  Kernel *result() override { return getContext()->get<Kernel>(); }

  STEP_NAME("create-axi-stream-mapper");
};

class AXI4StreamMapper : public firp::Module<AXI4StreamMapper> {
  axi4lite::AXI4LiteConfig liteConfig;
  axi4::AXI4Config writeConfig, readConfig;
  firp::axis::AXIStreamConfig mAxisConfig, sAxisControllerConfig, sAxisConfig, mAxisControllerConfig;
  circt::firrtl::FModuleOp ipecLoadUnit;
  circt::firrtl::FModuleOp ipecStoreUnit;
  circt::firrtl::FModuleOp spnAxisController;
public:
  AXI4StreamMapper(const axi4lite::AXI4LiteConfig& liteConfig,
                   const axi4::AXI4Config& writeConfig,
                   const axi4::AXI4Config& readConfig,
                   const firp::axis::AXIStreamConfig& mAxisConfig,
                   const firp::axis::AXIStreamConfig& sAxisControllerConfig,
                   const firp::axis::AXIStreamConfig& sAxisConfig,
                   const firp::axis::AXIStreamConfig& mAxisControllerConfig,
                   circt::firrtl::FModuleOp ipecLoadUnit,
                   circt::firrtl::FModuleOp ipecStoreUnit,
                   circt::firrtl::FModuleOp spnAxisController)
    : Module<AXI4StreamMapper>(
      "AXI4StreamMapper",
      {
        firp::Port("S_AXI_LITE", true, axi4lite::axi4LiteType(liteConfig)),
        firp::Port("M_AXI", true, axi4::axi4Type(writeConfig, readConfig)),
        firp::Port("M_AXIS", true, firp::axis::AXIStreamBundleType(mAxisConfig)),
        firp::Port("S_AXIS_CONTROLLER", true, firp::axis::AXIStreamBundleType(sAxisControllerConfig)),
        firp::Port("S_AXIS", true, firp::axis::AXIStreamBundleType(sAxisConfig)),
        firp::Port("M_AXIS_CONTROLLER", true, firp::axis::AXIStreamBundleType(mAxisControllerConfig)),
        firp::Port("interrupt", false, firp::bitType())
      },
      liteConfig, writeConfig, readConfig, mAxisConfig, sAxisControllerConfig, sAxisConfig, mAxisControllerConfig
    ), 
      liteConfig(liteConfig),
      writeConfig(writeConfig),
      readConfig(readConfig),
      mAxisConfig(mAxisConfig),
      sAxisControllerConfig(sAxisControllerConfig),
      sAxisConfig(sAxisConfig),
      mAxisControllerConfig(mAxisControllerConfig),
      ipecLoadUnit(ipecLoadUnit),
      ipecStoreUnit(ipecStoreUnit),
      spnAxisController(spnAxisController)
    { build(); }

  void body();

  static AXI4StreamMapper make(
    const FPGAKernel& kernel,
    circt::firrtl::FModuleOp ipecLoadUnit,
    circt::firrtl::FModuleOp ipecStoreUnit,
    circt::firrtl::FModuleOp spnAxisController
  );
};

}