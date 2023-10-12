//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "IPUSubtarget.h"
#include "llvm/MC/MCSchedule.h"

using namespace spnc;
using namespace llvm;

const llvm::SubtargetFeatureKV IPUFeatureKV[] = {};
const llvm::SubtargetSubTypeKV IPUSubTypeKV[] = {};

const llvm::MCWriteProcResEntry IPUWriteProcResTable[] = {};
const llvm::MCWriteLatencyEntry IPUWriteLatencyTable[] = {};
const llvm::MCReadAdvanceEntry IPUReadAdvanceTable[] = {};

IPUSubtarget::IPUSubtarget(const Triple &TT, StringRef CPU, StringRef TuneCPU, StringRef FS, const TargetMachine &TM)
    : TargetSubtargetInfo(TT, CPU, TuneCPU, FS, makeArrayRef(IPUFeatureKV, 0), makeArrayRef(IPUSubTypeKV, 0),
                          IPUWriteProcResTable, IPUWriteLatencyTable, IPUReadAdvanceTable, nullptr, nullptr, nullptr),
      TLInfo(TM, *this) {}