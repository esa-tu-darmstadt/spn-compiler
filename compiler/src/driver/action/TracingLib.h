//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_TRACINGLIB_H
#define SPNC_TRACINGLIB_H

#include <util/FileSystem.h>
#include <driver/Actions.h>

namespace spnc {

    using BitcodeFile = File<FileType::LLVM_BC>;

class TracingLib : public ActionWithOutput<BitcodeFile> {

    public:
        explicit TracingLib(BitcodeFile outputFile);

        BitcodeFile& execute() override;

    private:

        BitcodeFile outFile;

        bool cached = false;

    };
}



#endif //SPNC_TRACINGLIB_H
