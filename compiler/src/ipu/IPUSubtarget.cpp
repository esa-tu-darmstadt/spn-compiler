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