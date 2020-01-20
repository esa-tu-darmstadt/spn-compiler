//
// Created by lukas on 18.01.20.
//

#ifndef SPNC_CLANGKERNELLINKING_H
#define SPNC_CLANGKERNELLINKING_H

#include <driver/Actions.h>
#include <util/FileSystem.h>
#include <driver/Kernel.h>

namespace spnc {

    class ClangKernelLinking : public ActionSingleInput<ObjectFile, SharedObject> {

    public:

        ClangKernelLinking(ActionWithOutput<ObjectFile>& _input, SharedObject outputFile,
                const std::string& kernelFunctionName);

        SharedObject& execute() override;

    private:

        SharedObject outFile;

        bool cached = false;

        std::string kernelName;

    };

}



#endif //SPNC_CLANGKERNELLINKING_H
