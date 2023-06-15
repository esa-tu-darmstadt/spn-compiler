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

class CreateVivadoProject : public StepSingleInput<CreateVivadoProject, Kernel>, public StepWithResult<Kernel> {
  static constexpr uint64_t KERNEL_ID = 123;
  VivadoProjectConfig config;
public:
  explicit CreateVivadoProject(StepWithResult<Kernel>& input, const VivadoProjectConfig& config):
    StepSingleInput<CreateVivadoProject, Kernel>(input),
    config(config) {}

  ExecutionResult executeStep(Kernel *kernel);

  Kernel *result() override { return getContext()->get<Kernel>(); }

  STEP_NAME("create-ipxact")
private:
  ExecutionResult tapascoCompose();
  static void execShell(const std::vector<std::string>& cmd);
  static std::string execShellAndGetOutput(const std::vector<std::string>& cmd);
  static std::optional<std::string> grepBitstreamPath(const std::string& shellOutput);
};

}