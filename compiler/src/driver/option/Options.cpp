//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <driver/Options.h>

using namespace spnc::interface;

// Definitions of the static members of class Options.
std::unordered_map<std::string, Opt*> Options::options;
std::vector<std::unique_ptr<OptModifier>> Options::allModifiers;
std::vector<OptModifier*> Options::activeModifiers;
