#pragma once

#include "Kernel.h"
#include "config.hpp"

#include "pipeline/steps/mlir/MLIRPassPipeline.h"

#include "mlir/IR/BuiltinOps.h"
#include "pipeline/PipelineStep.h"
#include "toolchain/MLIRToolchain.h"

#include <firp/AXI4.hpp>
#include <firp/AXI4Lite.hpp>
#include <firp/AXIStream.hpp>
#include <firp/AXIStreamConverter.hpp>

namespace spnc {

class CreateAXIStreamMapper
    : public StepSingleInput<CreateAXIStreamMapper, mlir::ModuleOp>,
      public StepWithResult<mlir::ModuleOp> {
  //
  circt::firrtl::FModuleOp findModuleByName(const std::string &name);
  circt::firrtl::FModuleOp insertFIRFile(const std::filesystem::path &path,
                                         const std::string &moduleName);

  bool doPrepareForCocoTb;
  std::unique_ptr<mlir::ModuleOp> modOp;

public:
  explicit CreateAXIStreamMapper(StepWithResult<mlir::ModuleOp> &root,
                                 bool doPrepareForCocoTb)
      : StepSingleInput<CreateAXIStreamMapper, mlir::ModuleOp>(root),
        doPrepareForCocoTb(doPrepareForCocoTb) {}

  ExecutionResult executeStep(mlir::ModuleOp *root);

  mlir::ModuleOp *result() override { return modOp.get(); }

  STEP_NAME("create-axi-stream-mapper");
};

class AXI4StreamMapper : public firp::Module<AXI4StreamMapper> {
  axi4lite::AXI4LiteConfig liteConfig;
  axi4::AXI4Config writeConfig, readConfig;
  firp::axis::AXIStreamConfig mAxisConfig, sAxisControllerConfig, sAxisConfig,
      mAxisControllerConfig;
  circt::firrtl::FModuleOp ipecLoadUnit;
  circt::firrtl::FModuleOp ipecStoreUnit;
  circt::firrtl::FModuleOp spnAxisController;

public:
  AXI4StreamMapper(const axi4lite::AXI4LiteConfig &liteConfig,
                   const axi4::AXI4Config &writeConfig,
                   const axi4::AXI4Config &readConfig,
                   const firp::axis::AXIStreamConfig &mAxisConfig,
                   const firp::axis::AXIStreamConfig &sAxisControllerConfig,
                   const firp::axis::AXIStreamConfig &sAxisConfig,
                   const firp::axis::AXIStreamConfig &mAxisControllerConfig,
                   circt::firrtl::FModuleOp ipecLoadUnit,
                   circt::firrtl::FModuleOp ipecStoreUnit,
                   circt::firrtl::FModuleOp spnAxisController)
      : Module<AXI4StreamMapper>(
            "AXI4StreamMapper",
            {firp::Input("S_AXI_LITE", axi4lite::axi4LiteFlattenType(
                                           axi4lite::axi4LiteType(liteConfig))),
             firp::Output("M_AXI", axi4::axi4FlattenType(axi4::axi4Type(
                                       writeConfig, readConfig))),
             firp::Output("M_AXIS",
                          firp::axis::AXIStreamBundleType(mAxisConfig)),
             firp::Input("S_AXIS_CONTROLLER", firp::axis::AXIStreamBundleType(
                                                  sAxisControllerConfig)),
             firp::Input("S_AXIS",
                         firp::axis::AXIStreamBundleType(sAxisConfig)),
             firp::Output("M_AXIS_CONTROLLER", firp::axis::AXIStreamBundleType(
                                                   mAxisControllerConfig)),
             firp::Output("interrupt", firp::bitType())},
            liteConfig, writeConfig, readConfig, mAxisConfig,
            sAxisControllerConfig, sAxisConfig, mAxisControllerConfig),
        liteConfig(liteConfig), writeConfig(writeConfig),
        readConfig(readConfig), mAxisConfig(mAxisConfig),
        sAxisControllerConfig(sAxisControllerConfig), sAxisConfig(sAxisConfig),
        mAxisControllerConfig(mAxisControllerConfig),
        ipecLoadUnit(ipecLoadUnit), ipecStoreUnit(ipecStoreUnit),
        spnAxisController(spnAxisController) {
    build();
  }

  void body();

  static AXI4StreamMapper make(const FPGAKernel &kernel,
                               circt::firrtl::FModuleOp ipecLoadUnit,
                               circt::firrtl::FModuleOp ipecStoreUnit,
                               circt::firrtl::FModuleOp spnAxisController);
};

class AXI4StreamMapper_mimo : public firp::Module<AXI4StreamMapper_mimo> {
  axi4lite::AXI4LiteConfig liteConfig;
  axi4::AXI4Config writeConfig, readConfig;
  firp::axis::AXIStreamConfig mAxisConfig, sAxisControllerConfig, sAxisConfig,
      mAxisControllerConfig;
  circt::firrtl::FModuleOp ipecLoadUnit;
  circt::firrtl::FModuleOp ipecStoreUnit;
  circt::firrtl::FModuleOp spnAxisController;

public:
  AXI4StreamMapper_mimo(
      const axi4lite::AXI4LiteConfig &liteConfig,
      const axi4::AXI4Config &writeConfig, const axi4::AXI4Config &readConfig,
      const firp::axis::AXIStreamConfig &mAxisConfig,
      const firp::axis::AXIStreamConfig &sAxisControllerConfig,
      const firp::axis::AXIStreamConfig &sAxisConfig,
      const firp::axis::AXIStreamConfig &mAxisControllerConfig,
      circt::firrtl::FModuleOp ipecLoadUnit,
      circt::firrtl::FModuleOp ipecStoreUnit,
      circt::firrtl::FModuleOp spnAxisController)
      : Module<AXI4StreamMapper_mimo>(
            "AXI4StreamMapper",
            {firp::Input("S_AXI_LITE", axi4lite::axi4LiteFlattenType(
                                           axi4lite::axi4LiteType(liteConfig))),
             firp::Output("M_AXI", axi4::axi4FlattenType(axi4::axi4Type(
                                       writeConfig, readConfig))),
             // firp::Output("M_AXIS",
             // firp::axis::AXIStreamBundleType(mAxisConfig)),
             // firp::Input("S_AXIS_CONTROLLER",
             // firp::axis::AXIStreamBundleType(sAxisControllerConfig)),
             // firp::Input("S_AXIS",
             // firp::axis::AXIStreamBundleType(sAxisConfig)),
             // firp::Output("M_AXIS_CONTROLLER",
             // firp::axis::AXIStreamBundleType(mAxisControllerConfig)),
             firp::Output("interrupt", firp::bitType())},
            liteConfig, writeConfig, readConfig, mAxisConfig,
            sAxisControllerConfig, sAxisConfig, mAxisControllerConfig),
        liteConfig(liteConfig), writeConfig(writeConfig),
        readConfig(readConfig), mAxisConfig(mAxisConfig),
        sAxisControllerConfig(sAxisControllerConfig), sAxisConfig(sAxisConfig),
        mAxisControllerConfig(mAxisControllerConfig),
        ipecLoadUnit(ipecLoadUnit), ipecStoreUnit(ipecStoreUnit),
        spnAxisController(spnAxisController) {
    build();
  }

  void body();

  static AXI4StreamMapper_mimo make(const FPGAKernel &kernel,
                                    circt::firrtl::FModuleOp ipecLoadUnit,
                                    circt::firrtl::FModuleOp ipecStoreUnit,
                                    circt::firrtl::FModuleOp spnAxisController);
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
  AXI4CocoTbTop(const axi4lite::AXI4LiteConfig &liteConfig,
                const axi4::AXI4Config &writeConfig,
                const axi4::AXI4Config &readConfig,
                const firp::axis::AXIStreamConfig &mAxisConfig,
                const firp::axis::AXIStreamConfig &sAxisControllerConfig,
                const firp::axis::AXIStreamConfig &sAxisConfig,
                const firp::axis::AXIStreamConfig &mAxisControllerConfig,
                circt::firrtl::FModuleOp ipecLoadUnit,
                circt::firrtl::FModuleOp ipecStoreUnit,
                circt::firrtl::FModuleOp spnAxisController)
      : Module<AXI4CocoTbTop>(
            "AXI4CocoTbTop",
            {firp::Input("S_AXI_LITE", axi4lite::axi4LiteFlattenType(
                                           axi4lite::axi4LiteType(liteConfig))),
             firp::Output("M_AXI", axi4::axi4FlattenType(axi4::axi4Type(
                                       writeConfig, readConfig))),
             firp::Output("interrupt", firp::bitType())},
            liteConfig, writeConfig, readConfig),
        liteConfig(liteConfig), writeConfig(writeConfig),
        readConfig(readConfig), mAxisConfig(mAxisConfig),
        sAxisControllerConfig(sAxisControllerConfig), sAxisConfig(sAxisConfig),
        mAxisControllerConfig(mAxisControllerConfig),
        ipecLoadUnit(ipecLoadUnit), ipecStoreUnit(ipecStoreUnit),
        spnAxisController(spnAxisController) {
    build();
  }

  void body();

  static AXI4CocoTbTop make(const FPGAKernel &kernel,
                            circt::firrtl::FModuleOp ipecLoadUnit,
                            circt::firrtl::FModuleOp ipecStoreUnit,
                            circt::firrtl::FModuleOp spnAxisController);
};

class DummyWrapper : public firp::Module<DummyWrapper> {
  axi4lite::AXI4LiteConfig liteConfig;
  axi4::AXI4Config writeConfig;
  axi4::AXI4Config readConfig;

public:
  DummyWrapper(const axi4lite::AXI4LiteConfig &liteConfig,
               const axi4::AXI4Config &writeConfig,
               const axi4::AXI4Config &readConfig)
      : Module<DummyWrapper>(
            "DummyWrapper",
            {firp::Input("S_AXI_LITE", axi4lite::axi4LiteFlattenType(
                                           axi4lite::axi4LiteType(liteConfig))),
             firp::Output("M_AXI", axi4::axi4FlattenType(axi4::axi4Type(
                                       writeConfig, readConfig))),
             firp::Output("interrupt", firp::bitType())},
            liteConfig, writeConfig, readConfig),
        liteConfig(liteConfig), writeConfig(writeConfig),
        readConfig(readConfig) {
    build();
  }

  void body();

  static DummyWrapper make(const FPGAKernel &kernel);
};

class RegisterFile : public firp::Module<RegisterFile> {
  axi4lite::AXI4LiteConfig liteConfig;
  axi4::AXI4Config writeConfig;
  axi4::AXI4Config readConfig;

public:
  RegisterFile(const axi4lite::AXI4LiteConfig &liteConfig,
               const axi4::AXI4Config &writeConfig,
               const axi4::AXI4Config &readConfig)
      : Module<RegisterFile>(
            "RegisterFile",
            {firp::Input("S_AXI_LITE", axi4lite::axi4LiteFlattenType(
                                           axi4lite::axi4LiteType(liteConfig))),
             firp::Output("M_AXI", axi4::axi4FlattenType(axi4::axi4Type(
                                       writeConfig, readConfig))),
             firp::Output("interrupt", firp::bitType())},
            liteConfig, writeConfig, readConfig),
        liteConfig(liteConfig), writeConfig(writeConfig),
        readConfig(readConfig) {
    build();
  }

  void body();
};

} // namespace spnc