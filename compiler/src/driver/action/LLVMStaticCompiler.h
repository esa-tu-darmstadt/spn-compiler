//
// Created by ls on 1/16/20.
//

#ifndef SPNC_LLVMSTATICCOMPILER_H
#define SPNC_LLVMSTATICCOMPILER_H

#include <util/FileSystem.h>
#include <driver/Actions.h>

namespace spnc {

    using BitcodeFile = File<FileType::LLVM_BC>;
    using ObjectFile = File<FileType::OBJECT>;

    class LLVMStaticCompiler : public ActionSingleInput<BitcodeFile, ObjectFile> {

    public:
        explicit LLVMStaticCompiler(ActionWithOutput<BitcodeFile>& _input, ObjectFile outputFile);

        ObjectFile& execute() override;

    private:

        ObjectFile outFile;

        bool cached = false;

    };
}



#endif //SPNC_LLVMSTATICCOMPILER_H
