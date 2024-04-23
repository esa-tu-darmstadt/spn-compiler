#pragma once

#include "MLIRToolchain.h"
#include "pipeline/Pipeline.h"


namespace spnc {
    class FPGAToolchain : public MLIRToolchain {
      public:
        static std::unique_ptr<Pipeline<Kernel>> setupPipeline(const std::string& inputFile,
                                                               std::unique_ptr<interface::Configuration> config);
    };
}