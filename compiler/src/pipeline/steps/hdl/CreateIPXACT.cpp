#include "CreateIPXACT.h"

#include "spdlog/fmt/fmt.h"
#include "util/Command.h"
#include <fstream>


namespace spnc {

ExecutionResult CreateIPXACT::executeStep(std::string *verilogSource) {
  namespace fs = std::filesystem;

  fs::create_directory(config.targetDir);

  // copy source files into target dir
  for (const auto& from : config.sourceFilePaths) {
    std::error_code ec;
    fs::copy_file(from, config.targetDir, fs::copy_options::overwrite_existing, ec);

    if (!ec)
      return failure(
        fmt::format("File {} could not be copied to {}", from.string(), config.targetDir.string())
      );
  };

  // write verilog source to target dir
  fs::path path = config.targetDir / config.topModuleFileName;
  std::ofstream outFile(path);

  if (!outFile.is_open())
    return failure(
      fmt::format("Could not open file {}", path.string())
    );

  outFile << *verilogSource;
  outFile.close();

  // TODO: Build IPXACT XMLs
  return failure("CreateIPXACT cannot return a kernel!");
}

}