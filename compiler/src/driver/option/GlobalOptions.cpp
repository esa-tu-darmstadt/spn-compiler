//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <driver/GlobalOptions.h>
#include <driver/Options.h>

using namespace spnc::interface;

Option<bool> spnc::option::collectGraphStats{"collect-graph-stats", false};

Option<std::string> spnc::option::graphStatsFile{"graph-stats-file",
                                                 "/tmp/stats.json",
                                                 {depends(spnc::option::collectGraphStats, true)}};

using spnc::option::TargetMachine;
EnumOpt spnc::option::compilationTarget{"target",
                                        {EnumVal(CPU, "CPU")},
                                        {required()}};

Option<bool> spnc::option::cpuVectorize{"cpu-vectorize", false};

Option<bool> spnc::option::deleteTemporaryFiles{"delete-temps", true};

Option<bool> spnc::option::useMLIRToolchain{"use-mlir", true};