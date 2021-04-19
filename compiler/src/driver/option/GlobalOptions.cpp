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
                                        {EnumVal(CPU, "CPU"),
                                         EnumVal(CUDA, "CUDA")},
                                        {required()}};

Option<bool> spnc::option::cpuVectorize{"cpu-vectorize", false};

using spnc::option::VectorLibrary;
EnumOpt spnc::option::vectorLibrary{"vector-library", NONE,
                                    {EnumVal(SVML, "SVML"),
                                     EnumVal(LIBMVEC, "LIBMVEC"),
                                     EnumVal(NONE, "None")}};

Option<bool> spnc::option::replaceGatherWithShuffle{"use-shuffle", false};

Option<bool> spnc::option::logSpace{"use-log-space", false};

Option<bool> spnc::option::gpuSharedMem{"gpu-shared-mem", true};

Option<std::string> spnc::option::searchPaths{"search-paths", ""};

Option<bool> spnc::option::deleteTemporaryFiles{"delete-temps", true};

Option<bool> spnc::option::dumpIR{"dump-ir", true};

Option<bool> spnc::option::optRepresentation{"opt-repr", false};

using spnc::option::RepresentationOption;
EnumOpt spnc::option::representationFormat{"opt-repr-format",
                                           {EnumVal(FLOATING_POINT, "floating point representation"),
                                            EnumVal(FIXED_POINT, "fixed point representation")},
                                           {depends(spnc::option::optRepresentation, true)}};

using spnc::option::RepresentationErrorKind;
EnumOpt spnc::option::optimalRepresentationErrorKind{"opt-repr-err-kind",
                                                     {EnumVal(ERR_ABSOLUTE, "absolute error"),
                                                      EnumVal(ERR_RELATIVE, "relative error"),
                                                      EnumVal(ERR_ABS_LOG, "absolute logarithmic error")},
                                                      {depends(spnc::option::optRepresentation, true)}};

Option<double> spnc::option::optimalRepresentationErrorThreshold{"opt-repr-err-threshold",
                                                                 0.1,
                                                                 {depends(spnc::option::optRepresentation, true)}};
