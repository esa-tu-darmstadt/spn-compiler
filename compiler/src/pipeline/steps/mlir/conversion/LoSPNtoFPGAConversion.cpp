#include "LoSPNtoFPGAConversion.h"

#include "LoSPNtoFPGA/LoSPNtoFPGAPass.h"
#include "LoSPNtoFPGA2/LoSPNtoFPGAPass2.hpp"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Dialect/HW/HWOps.h"

#include <filesystem>
#include <util/json.hpp>
#include <Kernel.h>


namespace spnc {

void LoSPNtoFPGAConversion::preProcess(mlir::ModuleOp *inputModule) {
  namespace fs = std::filesystem;
  using json = nlohmann::json;

  if (fpgaConfigJson.empty()) {
    spdlog::info("No fpga-config-json file specified!");
    return;
  }

  std::string jsonText;

  if (fs::is_regular_file(fpgaConfigJson)) {
    spdlog::info("Attempting to parse fpga-config-json from file {}", fpgaConfigJson);
    std::ifstream jsonFile(fpgaConfigJson);
    assert(jsonFile.is_open() && "Could not open fpga-config-json file");
    jsonFile >> jsonText;
  } else {
    spdlog::info("Attempting to parse fpga-config-json from command line argument");
    jsonText = fpgaConfigJson;
  }

  FPGAKernel& kernel = getContext()->get<Kernel>()->getFPGAKernel();

  // Don't ask me what happens when an invalid json is passed. Exceptions are disabled Q_Q
  json j = json::parse(jsonText);

  kernel.memDataWidth = j.at("axi4").at("dataWidth");
  kernel.memAddrWidth = j.at("axi4").at("addrWidth");

  kernel.liteDataWidth = j.at("axi4lite").at("dataWidth");
  kernel.liteAddrWidth = j.at("axi4lite").at("addrWidth");

  kernel.kernelName = j.at("kernelName");
  kernel.kernelId = j.at("kernelId");
}

void LoSPNtoFPGAConversion::initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx) {

  pm->addPass(
    mlir::spn::fpga::createLoSPNtoFPGAPass2()
  );

  //struct LowerSeqFIRRTLToSVOptions {
  //  bool disableRegRandomization = false;
  //  bool addVivadoRAMAddressConflictSynthesisBugWorkaround = false;
  //};
  circt::seq::LowerSeqFIRRTLToSVOptions options;

  pm->nest<circt::hw::HWModuleOp>().addPass(
    circt::seq::createSeqFIRRTLLowerToSVPass(options)
  );

}

void LoSPNtoFPGAConversion::postProcess(mlir::ModuleOp *resultModule) {
  FPGAKernel& kernel = getContext()->get<Kernel>()->getFPGAKernel();

  resultModule->walk([&](mlir::Operation *op){
    if (!op->hasAttr("fpga.body_delay"))
      return;

    kernel.bodyDelay = llvm::dyn_cast<mlir::IntegerAttr>(
      op->getAttr("fpga.body_delay")
    ).getInt();
  });

  kernel.fifoDepth = kernel.bodyDelay * 2;

  spdlog::info(kernel.to_string());
}

}