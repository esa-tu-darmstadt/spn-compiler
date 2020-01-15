//
// Created by ls on 1/15/20.
//

#ifndef SPNC_CPUTOOLCHAIN_H
#define SPNC_CPUTOOLCHAIN_H

#include <llvm/IR/Module.h>
#include <driver/Job.h>

namespace spnc {

    class CPUToolchain {

    public:
        static std::unique_ptr<Job<llvm::Module>> constructJob(const std::string& inputFile) ;

    };

}



#endif //SPNC_CPUTOOLCHAIN_H
