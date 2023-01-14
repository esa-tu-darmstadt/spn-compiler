#include "LoSPNtoFPGAConversion.h"

#include "LoSPNtoFPGA/LoSPNtoFPGAPass.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Dialect/HW/HWOps.h"


namespace spnc {

void LoSPNtoFPGAConversion::initializePassPipeline(mlir::PassManager* pm, mlir::MLIRContext* ctx) {

  pm->addPass(
    mlir::spn::fpga::createLoSPNtoFPGAPass()
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

}