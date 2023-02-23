#pragma once

#include "spdlog/fmt/fmt.h"
#include "pipeline/PipelineStep.h"

#include <filesystem>


namespace spnc {

static const char JSON_TEMPLATE[] = R"(
{{
  "varCount": {varCount},
  "bitsPerVar": {bitsPerVar},
  "bodyDelay": {bodyDelay}
}}
)";

class WriteDebugInfo : public StepSingleInput<WriteDebugInfo, Kernel>,
                       public StepWithResult<Kernel> {
  fs::path targetPath;
public:
  explicit WriteDebugInfo(const fs::path& targetPath, StepWithResult<Kernel>& kernel):
    StepSingleInput<WriteDebugInfo, Kernel>(kernel),
    targetPath(targetPath) {}

  ExecutionResult executeStep(Kernel *kernel) {
    fs::create_directories(targetPath);

    fs::path targetFile = targetPath / "debug_info.json";
    std::ofstream outFile(targetFile);

    if (!outFile.is_open())
      return failure(
        fmt::format("could not open file {}", targetFile.string())
      );

    outFile << fmt::format(JSON_TEMPLATE,
      fmt::arg("varCount", kernel->getFPGAKernel().spnVarCount),
      fmt::arg("bitsPerVar", kernel->getFPGAKernel().spnBitsPerVar),
      fmt::arg("bodyDelay", kernel->getFPGAKernel().bodyDelay)
    );

    return success();
  }

  Kernel *result() override { return getContext()->get<Kernel>(); }

  STEP_NAME("write-debug-info")
};

}