//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================
#pragma once

#include "TargetExecutionModel.h"
namespace spnc {
class IPUTargetExecutionModel : public spnc::TargetExecutionModel {
    public:
    std::string getTargetName() const override { return "IPU"; }
  // TODO: Implement IPU-specific target execution model
};
} // namespace spnc