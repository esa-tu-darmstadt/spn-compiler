#include "CreateIPXACT.h"

#include "spdlog/fmt/fmt.h"
#include "util/Command.h"
#include "toolchain/MLIRToolchain.h"
#include <fstream>


namespace spnc {

// taken from https://github.com/esa-tu-darmstadt/BSVTools/blob/d4b389170e973b87b592a53896bb051fe5fe773e/scripts/bsvTools.py
static const char TCL[] = R"(
ipx::infer_core -vendor {vendor} -name {projectname} -library user -taxonomy /UserIP -files {directory}/src/{topModule}.v -root_dir {directory}
ipx::edit_ip_in_project -upgrade true -name edit_ip_project -directory {tmpdir} {directory}/component.xml
ipx::current_core {directory}/component.xml
set_property top {topModule} [current_fileset]
set_property -quiet interface_mode monitor [ipx::get_bus_interfaces *MON* -of_objects [ipx::current_core]]
add_files {directory}/src
update_compile_order -fileset sources_1
set_property name {projectname} [ipx::current_core]
set_property display_name {projectname} [ipx::current_core]
set_property description {projectname} [ipx::current_core]
set_property core_revision 1 [ipx::current_core]
set_property AUTO_FAMILY_SUPPORT_LEVEL level_1 [ipx::current_core]
foreach f {{ {includes} }} {{
    set_property is_global_include true [get_files $f]
}}
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1
ipx::merge_project_changes files [ipx::current_core]
ipx::merge_project_changes ports [ipx::current_core]
puts "USED FILES"
foreach f [ipx::get_files -of_objects [ipx::get_file_groups *synthesis*]] {{
    set n [get_property NAME $f]
    puts "USED FILE:$n"
}}
puts "END USED FILES"
puts "Additional Parameters"
{additional_parameters}
puts "End Additional Parameters"
ipx::create_xgui_files [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::save_core [ipx::current_core]
close_project -delete
puts "VIVADO FINISHED SUCCESSFULLY"
)";

ExecutionResult CreateIPXACT::executeStep(std::string *verilogSource) {
  namespace fs = std::filesystem;

  fs::create_directory(config.targetDir);
  fs::create_directory(config.targetDir / config.tmpdir);
  fs::create_directory(config.targetDir / "src");
  fs::path srcPath = config.targetDir / "src";

  // copy source files into target dir
  for (const auto& from : config.sourceFilePaths) {
    try {
      //llvm::outs() << "copying " << from.string() << " to " << config.targetDir.string() << "\n";
      fs::copy(from, srcPath, fs::copy_options::overwrite_existing);
    } catch (const fs::filesystem_error& err) {
      return failure(
        fmt::format("File {} could not be copied to {}: {}", from.string(), srcPath.string(), err.what())
      );
    }
  };

  // write verilog source to target dir
  {
    fs::path path = srcPath / config.topModuleFileName;
    std::ofstream outFile(path);

    if (!outFile.is_open())
      return failure(
        fmt::format("Could not open file {}", path.string())
      );

    outFile << *verilogSource;
  }

  // write simulation helper code to file
  {
    fs::path path = srcPath / "simimpl.cpp";
    std::ofstream outFile(path);

    if (!outFile.is_open())
      return failure(
        fmt::format("Could not open file {}", path.string())
      );

    outFile << generateSimulationSourceCode();
  }

  fs::path tclPath = config.targetDir / "build.tcl";

  // construct tcl script
  {
    std::string tclSource = fmt::format(TCL,
      fmt::arg("vendor", config.vendor),
      fmt::arg("projectname", config.projectName),
      fmt::arg("directory", config.directory),
      fmt::arg("topModule", config.topModule),
      fmt::arg("tmpdir", config.tmpdir),
      fmt::arg("includes", ""),
      fmt::arg("additional_parameters", "")
    );
    
    std::ofstream outFile(tclPath);

    if (!outFile.is_open())
      return failure(
        fmt::format("Could not open file {}", tclPath.string())
      );

    outFile << tclSource;
  }

  // call vivado
  {
    auto pwd = std::filesystem::current_path();
    std::filesystem::current_path(config.targetDir);

    std::vector<std::string> cmd{
      "vivado", "-mode", "batch", "-source", tclPath.string(), "-nojournal", "-nolog"
    };

    try {
      Command::executeExternalCommand(cmd);
    } catch (const std::runtime_error& e) {
      std::filesystem::current_path(pwd);
      spdlog::warn(e.what());
      spdlog::warn("Continuing without executing Vivado!");
    }
  }

  const KernelInfo *kernelInfo = getContext()->get<KernelInfo>();
  kernel = std::make_unique<Kernel>(
    "",
    "",
    kernelInfo->queryType, // query_type
    kernelInfo->target, //target
    kernelInfo->batchSize, // batch size
    kernelInfo->numFeatures, // num features
    kernelInfo->bytesPerFeature, // bytes per feature
    kernelInfo->numResults, // num results
    kernelInfo->bytesPerResult, // bytes per result
    "uint32_t" // data type
  );

  return success();
}

std::string CreateIPXACT::generateSimulationSourceCode() {
  KernelInfo kernelInfo = *getContext()->get<KernelInfo>();

  std::stringstream ss;

  // code for type conversion
  ss << R"(
#include "common.hpp"


uint8_t PySPNSim::convertIndex(uint32_t input) {
  uint8_t result = static_cast<uint8_t>(input);
  return result;
}

double PySPNSim::convertProb(uint32_t prob) {
  uint64_t p = prob;
  return *reinterpret_cast<const float *>(&p);
}
  )";

  // code for setting input
  std::stringstream setInputs;
  for (unsigned int i = 0; i < kernelInfo.numFeatures; ++i)
    setInputs << "  top->in_" << i << " = convertIndex(input[" << i << "]);\n";

  std::string txt = fmt::format(R"(
void PySPNSim::setInput(const std::vector<uint32_t>& input) {{
  assert(input.size() == {count});

{setInputs}
}}
  )",
    fmt::arg("count", kernelInfo.numFeatures),
    fmt::arg("setInputs", setInputs.str()));

  ss << txt;

  return ss.str();
}

}