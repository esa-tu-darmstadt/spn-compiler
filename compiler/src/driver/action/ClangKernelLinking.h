//
// Created by lukas on 18.01.20.
//

#ifndef SPNC_CLANGKERNELLINKING_H
#define SPNC_CLANGKERNELLINKING_H

#include <driver/Actions.h>
#include <util/FileSystem.h>
#include "../../../../common/include/Kernel.h"

namespace spnc {

    class ClangKernelLinking : public ActionSingleInput<ObjectFile, Kernel> {

    public:

        ClangKernelLinking(ActionWithOutput<ObjectFile>& _input, SharedObject outputFile,
                const std::string& kernelFunctionName);

        Kernel& execute() override;

    private:

        SharedObject outFile;

        std::string kernelName;

        Kernel kernel;

        bool cached = false;

    };

}



#endif //SPNC_CLANGKERNELLINKING_H
