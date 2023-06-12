#pragma once

#include "Kernel.h"

#include "pipeline/steps/mlir/MLIRPassPipeline.h"

#include "pipeline/PipelineStep.h"
#include "toolchain/MLIRToolchain.h"
#include "mlir/IR/BuiltinOps.h"

#include <firp/AXI4.hpp>
#include <firp/AXI4Lite.hpp>
#include <firp/AXIStream.hpp>
#include <firp/AXIStreamConverter.hpp>


namespace spnc {

class CreateAXIStreamMapper : public StepSingleInput<CreateAXIStreamMapper, mlir::ModuleOp>,
                              public StepWithResult<mlir::ModuleOp> {
  //
  circt::firrtl::FModuleOp findModuleByName(const std::string& name);
  circt::firrtl::FModuleOp insertFIRFile(const std::filesystem::path& path, const std::string& moduleName);

  bool doPrepareForCocoTb = true;
  std::unique_ptr<mlir::ModuleOp> modOp;
public:
  explicit CreateAXIStreamMapper(StepWithResult<mlir::ModuleOp>& root):
    StepSingleInput<CreateAXIStreamMapper, mlir::ModuleOp>(root) {}

  ExecutionResult executeStep(mlir::ModuleOp *root);

  mlir::ModuleOp *result() override { return modOp.get(); }

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
        firp::Port("M_AXI", false, axi4::axi4Type(writeConfig, readConfig)),
        firp::Port("M_AXIS", false, firp::axis::AXIStreamBundleType(mAxisConfig)),
        firp::Port("S_AXIS_CONTROLLER", true, firp::axis::AXIStreamBundleType(sAxisControllerConfig)),
        firp::Port("S_AXIS", true, firp::axis::AXIStreamBundleType(sAxisConfig)),
        firp::Port("M_AXIS_CONTROLLER", false, firp::axis::AXIStreamBundleType(mAxisControllerConfig)),
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

class AXI4CocoTbTop : public firp::Module<AXI4CocoTbTop> {
  axi4lite::AXI4LiteConfig liteConfig;
  axi4::AXI4Config writeConfig;
  axi4::AXI4Config readConfig;
  firp::axis::AXIStreamConfig mAxisConfig;
  firp::axis::AXIStreamConfig sAxisControllerConfig;
  firp::axis::AXIStreamConfig sAxisConfig;
  firp::axis::AXIStreamConfig mAxisControllerConfig;
  circt::firrtl::FModuleOp ipecLoadUnit;
  circt::firrtl::FModuleOp ipecStoreUnit;
  circt::firrtl::FModuleOp spnAxisController;
public:
  AXI4CocoTbTop(const axi4lite::AXI4LiteConfig& liteConfig,
                const axi4::AXI4Config& writeConfig,
                const axi4::AXI4Config& readConfig,
                const firp::axis::AXIStreamConfig& mAxisConfig,
                const firp::axis::AXIStreamConfig& sAxisControllerConfig,
                const firp::axis::AXIStreamConfig& sAxisConfig,
                const firp::axis::AXIStreamConfig& mAxisControllerConfig,
                circt::firrtl::FModuleOp ipecLoadUnit,
                circt::firrtl::FModuleOp ipecStoreUnit,
                circt::firrtl::FModuleOp spnAxisController)
    : Module<AXI4CocoTbTop>(
      "AXI4CocoTbTop",
      {
        firp::Port("S_AXI_LITE", true, axi4lite::axi4LiteType(liteConfig)),
        firp::Port("M_AXI", false, axi4::axi4Type(writeConfig, readConfig))
      },
      liteConfig, writeConfig, readConfig
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

  static AXI4CocoTbTop make(
    const FPGAKernel& kernel,
    circt::firrtl::FModuleOp ipecLoadUnit,
    circt::firrtl::FModuleOp ipecStoreUnit,
    circt::firrtl::FModuleOp spnAxisController
  );
};

}