//
// Created by ls on 1/15/20.
//

#ifndef SPNC_LLVMWRITEBITCODE_H
#define SPNC_LLVMWRITEBITCODE_H


#include <llvm/IR/Module.h>
#include <driver/BaseActions.h>

namespace spnc {

    class LLVMWriteBitcode : public ActionSingleInput<llvm::Module, File<FileType::LLVM_BC>> {

    public:

        explicit LLVMWriteBitcode(ActionWithOutput<llvm::Module>& _input, const std::string& outputFile);

        File<FileType::LLVM_BC> &execute() override;

    private:

        File<FileType::LLVM_BC> outFile;

        bool cached = false;

    };
}



#endif //SPNC_LLVMWRITEBITCODE_H
