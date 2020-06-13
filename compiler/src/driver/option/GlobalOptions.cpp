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


Option<bool> spnc::option::deleteTemporaryFiles{"delete-temps", true};

Option<bool> spnc::option::forceTree{"forceTree", false};

using spnc::option::BodyCGMethod;
EnumOpt spnc::option::bodyCodeGenMethod("bodyCodeGenMethod", Scalar, {EnumVal(Scalar, "Scalar"), EnumVal(ILP, "ILP"), EnumVal(Heuristic, "Heuristic")});

Option<int> spnc::option::simdWidth{"simdWidth", 2};

Option<bool> spnc::option::useGather{"useGather", false};
Option<bool> spnc::option::selectBinary{"selectBinary", false};
Option<bool> spnc::option::incSolve{"iterativeSolving", false};

Option<int> spnc::option::rootCand{"rootCand", 20};
Option<int> spnc::option::depCand{"depCand", 5};
Option<int> spnc::option::chainCandidates{"chainCandidates", 500};
Option<int> spnc::option::depChains{"depChains", 10};
