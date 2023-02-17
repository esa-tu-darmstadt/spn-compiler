#pragma once

#include "pipeline/PipelineStep.h"
#include "mlir/IR/BuiltinOps.h"
#include "Kernel.h"

#include <filesystem>


namespace spnc {

struct VivadoProjectConfig {
  std::vector<std::filesystem::path> sourceFilePaths;
  std::filesystem::path targetDir;
  std::filesystem::path topModuleFileName;

  std::string vendor = "esa.informatik.tu-darmstadt.de";
  std::string projectName = "spnc";
  std::string directory = ".";
  std::string topModule = "spn_body";
  std::string tmpdir = "tmp";

  std::string version = "1.23";
  std::string device = "ultra96v2";
  uint32_t mhz = 200;

  void addSourceFilePath(const std::filesystem::path& path);
};

class CreateVivadoProject : public StepSingleInput<CreateVivadoProject, mlir::ModuleOp>,
                     // TODO: Make the compiler not expect a Kernel as the pipeline result.
                     public StepWithResult<Kernel> {
  static constexpr uint64_t KERNEL_ID = 123;
  VivadoProjectConfig config;
public:
  explicit CreateVivadoProject(const VivadoProjectConfig& config, StepWithResult<mlir::ModuleOp>& input):
    StepSingleInput<CreateVivadoProject, mlir::ModuleOp>(input),
    config(config) {}

  ExecutionResult executeStep(mlir::ModuleOp *mod);

  Kernel *result() override { return kernel.get(); }

  STEP_NAME("create-ipxact")
private:
  std::unique_ptr<Kernel> kernel;

  ExecutionResult tapascoCompose();
  static void execShell(const std::vector<std::string>& cmd);
};

}