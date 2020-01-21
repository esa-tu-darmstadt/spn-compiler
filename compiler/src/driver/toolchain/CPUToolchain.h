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
        static std::unique_ptr<Job<SharedObject>> constructJobFromFile(const std::string& inputFile) ;

        static std::unique_ptr<Job<SharedObject>> constructJobFromString(const std::string& inputString) ;

    private:
        static std::unique_ptr<Job<SharedObject>> constructJob(std::unique_ptr<ActionWithOutput<std::string>> input);

    };

}



#endif //SPNC_CPUTOOLCHAIN_H
