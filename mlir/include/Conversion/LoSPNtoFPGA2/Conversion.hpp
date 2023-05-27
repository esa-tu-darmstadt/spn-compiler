#pragma once


#include <unordered_map>

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"

#include "circt/Dialect/HWArith/HWArithDialect.h"
#include "circt/Dialect/HWArith/HWArithOps.h"
#include "circt/Dialect/HWArith/HWArithTypes.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"

#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"
#include "LoSPN/LoSPNAttributes.h"
#include "HiSPN/HiSPNDialect.h"

#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Problems.h"
#include "circt/Scheduling/Utilities.h"

#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"

//#include "scheduling.hpp"
//#include "types.hpp"
#include <firp/ufloat.hpp>


namespace mlir::spn::fpga {

struct ConversionOptions {
  ufloat::UFloatConfig ufloatConfig;
  circt::firrtl::FIRRTLBaseType probType;
  circt::firrtl::FIRRTLBaseType indexType;
  bool use32Bit;
};

llvm::Optional<mlir::ModuleOp> convert(mlir::ModuleOp modOp, const ConversionOptions& options);


}
