//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_TRACINGLIB_H
#define SPNC_TRACINGLIB_H

#include <memory>
#include <util/FileSystem.h>
#include <driver/Actions.h>

namespace spnc {

    using BitcodeFile = File<FileType::LLVM_BC>;

class DetectTracingLib : public ActionWithOutput<BitcodeFile> {

    public:
        explicit DetectTracingLib();

        BitcodeFile& execute() override;

    private:

        std::unique_ptr<BitcodeFile> outFile;

        bool cached = false;

        bool error = false;

    };
}



#endif //SPNC_TRACINGLIB_H
