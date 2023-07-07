#pragma once

#include "spdlog/fmt/fmt.h"
#include "pipeline/PipelineStep.h"

#include <filesystem>
#include <util/json.hpp>


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
  std::string fpgaConfigJson;
public:
  explicit WriteDebugInfo(const fs::path& targetPath, const std::string& fpgaConfigJson, StepWithResult<Kernel>& kernel):
    StepSingleInput<WriteDebugInfo, Kernel>(kernel),
    targetPath(targetPath), fpgaConfigJson(fpgaConfigJson) {}

  ExecutionResult executeStep(Kernel *kernel) {
    namespace fs = std::filesystem;
    using json = nlohmann::json;

    fs::create_directories(targetPath);

    std::string jsonText;

    if (fs::is_regular_file(fpgaConfigJson)) {
      spdlog::info("Attempting to parse fpga-config-json from file {}", fpgaConfigJson);
      std::ifstream jsonFile(fpgaConfigJson);
      assert(jsonFile.is_open() && "Could not open fpga-config-json file");
      std::stringstream buffer;
      buffer << jsonFile.rdbuf();
      jsonText = buffer.str();
    } else {
      spdlog::info("Attempting to parse fpga-config-json from command line argument");
      jsonText = fpgaConfigJson;
    }

    // Don't ask me what happens when an invalid json is passed. Exceptions are disabled Q_Q
    json j = json::parse(jsonText);

    auto fpgaKernel = kernel->getFPGAKernel();

    j["varCount"] = fpgaKernel.spnVarCount;
    j["bitsPerVar"] = fpgaKernel.spnBitsPerVar;
    j["bodyDelay"] = fpgaKernel.bodyDelay;
    j["floatType"] = fpgaKernel.spnResultWidth == 32 ? "float32" : "float64";
    j["projectName"] = fpgaKernel.projectName;

    fs::path targetFile = targetPath / "config.json";
    std::ofstream outFile(targetFile);

    if (!outFile.is_open())
      return failure(
        fmt::format("could not open file {}", targetFile.string())
      );

    outFile << j.dump();

    return success();
  }

  Kernel *result() override { return getContext()->get<Kernel>(); }

  STEP_NAME("write-debug-info")
};

}