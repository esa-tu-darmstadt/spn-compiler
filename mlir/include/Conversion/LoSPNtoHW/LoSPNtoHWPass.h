#ifndef SPNS_MLIR_INCLUDE_CONVERSION_LOSPNTOHW_LOSPNTOHWPASS_H
#define SPNS_MLIR_INCLUDE_CONVERSION_LOSPNTOHW_LOSPNTOHWPASS_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/IR/PatternMatch.h"


namespace mlir
{
namespace spn
{
namespace hw
{



struct LoSPNtoHWPass : public PassWrapper<LoSPNtoHWPass, OperationPass<ModuleOp>>
{
public:
    LoSPNtoHWPass() = default;
    StringRef getArgument() const override { return "convert-lospn-to-hw"; }
    StringRef getDescription() const override { return "lalalal"; }

    void convertManual();
protected:
    void runOnOperation() override
    {
        llvm::errs() << "Not implemented!\n";
    }
};

}
}
}

#endif // SPNS_MLIR_INCLUDE_CONVERSION_LOSPNTOHW_LOSPNTOHWPASS_H