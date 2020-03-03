//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_LLVMLINKER_H
#define SPNC_LLVMLINKER_H

#include <util/FileSystem.h>
#include <driver/Actions.h>

namespace spnc {

    using BitcodeFile = File<FileType::LLVM_BC>;

    class LLVMLinker : public ActionDualInput<BitcodeFile, BitcodeFile, BitcodeFile> {

    public:
        explicit LLVMLinker(ActionWithOutput<BitcodeFile>& _input1,
                            ActionWithOutput<BitcodeFile>& _input2,
                            BitcodeFile outputFile);

        BitcodeFile& execute() override;

    private:

        BitcodeFile outFile;

        bool cached = false;

    };
}



#endif //SPNC_LLVMLINKER_H
