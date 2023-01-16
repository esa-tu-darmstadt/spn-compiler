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
  std::string projectName = "test.project";
  std::string directory = ".";
  std::string topModule = "spn_body";
  std::string tmpdir = "tmp";

  void addSourceFilePath(const std::filesystem::path& path);
};

class CreateIPXACT : public StepSingleInput<CreateIPXACT, std::string>,
                     // TODO: Make the compiler not expect a Kernel as the pipeline result.
                     public StepWithResult<Kernel> {
  IPXACTConfig config;
public:
  explicit CreateIPXACT(const IPXACTConfig& config, StepWithResult<std::string>& input):
    StepSingleInput<CreateIPXACT, std::string>(input),
    config(config) {}

  ExecutionResult executeStep(std::string *verilogSource);

  Kernel *result() override { return kernel.get(); }

  STEP_NAME("create-ipxact")
private:
  std::unique_ptr<Kernel> kernel = std::make_unique<Kernel>(
    "", "", 0, 0, 0, 0, 0, 0, 0, ""
  );
};

}