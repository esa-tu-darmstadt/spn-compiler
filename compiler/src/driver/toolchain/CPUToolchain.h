//
// Created by ls on 1/15/20.
//

#ifndef SPNC_CPUTOOLCHAIN_H
#define SPNC_CPUTOOLCHAIN_H

#include <llvm/IR/Module.h>
#include <driver/Job.h>
#include <driver/BaseActions.h>

namespace spnc {

    class CPUToolchain {

    public:
        static std::unique_ptr<Job<SharedObject>> constructJob(const std::string& inputFile) ;

    };

}



#endif //SPNC_CPUTOOLCHAIN_H
