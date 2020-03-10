//
// Created by ls on 1/15/20.
//

#ifndef SPNC_CPUTOOLCHAIN_H
#define SPNC_CPUTOOLCHAIN_H

#include <llvm/IR/Module.h>
#include <driver/Job.h>
#include <driver/BaseActions.h>
#include <driver/Options.h>
#include "../../../../common/include/Kernel.h"

using namespace spnc::interface;

namespace spnc {

  class CPUToolchain {

  public:
    static std::unique_ptr<Job<Kernel>> constructJobFromFile(const std::string& inputFile, const Configuration& config);

    static std::unique_ptr<Job<Kernel>> constructJobFromString(const std::string& inputString,
                                                               const Configuration& config);

    private:
    static std::unique_ptr<Job<Kernel>> constructJob(std::unique_ptr<ActionWithOutput<std::string>> input,
                                                     const Configuration& config);

    };

}



#endif //SPNC_CPUTOOLCHAIN_H
