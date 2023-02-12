#pragma once

#include "pipeline/PipelineStep.h"
#include "mlir/IR/BuiltinOps.h"
#include "Kernel.h"

#include <filesystem>


namespace spnc {

struct IPXACTConfig {
  std::vector<std::filesystem::path> sourceFilePaths;
  std::filesystem::path targetDir;
  std::filesystem::path topModuleFileName;

  std::string vendor = "esa.informatik.tu-darmstadt.de";
  std::string projectName = "spnc";
  std::string directory = ".";
  std::string topModule = "spn_body";
  std::string tmpdir = "tmp";

  uint32_t liteAddrWidth;
  uint32_t liteDataWidth;
  uint32_t mmAddrWidth;
  uint32_t mmDataWidth;

  void addSourceFilePath(const std::filesystem::path& path);
};

class CreateIPXACT : public StepSingleInput<CreateIPXACT, mlir::ModuleOp>,
                     // TODO: Make the compiler not expect a Kernel as the pipeline result.
                     public StepWithResult<Kernel> {
  IPXACTConfig config;
public:
  explicit CreateIPXACT(const IPXACTConfig& config, StepWithResult<mlir::ModuleOp>& input):
    StepSingleInput<CreateIPXACT, mlir::ModuleOp>(input),
    config(config) {}

  ExecutionResult executeStep(mlir::ModuleOp *mod);

  Kernel *result() override { return kernel.get(); }

  STEP_NAME("create-ipxact")
private:
  std::unique_ptr<Kernel> kernel;

  std::string generateSimulationSourceCode();
  std::string generateFullSimulationSourceCode() const;

  static std::optional<std::string> getFileListString(const std::filesystem::path& fileListFile);
};

}