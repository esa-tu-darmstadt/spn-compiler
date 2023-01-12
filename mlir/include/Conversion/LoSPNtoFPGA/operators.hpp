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
#include "HiSPN/HiSPNDialect.h"

#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Problems.h"
#include "circt/Scheduling/Utilities.h"

#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"


namespace mlir::spn::fpga::operators {

enum OperatorType {
  // these come from external verilog sources
  TYPE_ADD, TYPE_MUL, TYPE_LOG,
  // these are generated
  TYPE_CATEGORICAL, TYPE_HISTOGRAM, TYPE_CONSTANT,
  // not important for scheduling
  TYPE_YIELD
};

// Since everything becomes a InstanceOp or similar primitive HW type it
// is useful to track the original operator types.
class OperatorTypeMapping {
  std::unordered_map<Operation *, OperatorType> operatorTypes;
  std::unordered_map<OperatorType, std::string> baseNames;
  std::unordered_map<std::string, OperatorType> baseNameToType;
  std::unordered_map<OperatorType, uint64_t> delays;

  bool isExternalType(OperatorType type) const {
    return type == TYPE_ADD || type == TYPE_MUL || type == TYPE_LOG;
  }
public:
  OperatorTypeMapping() {
    // TODO: Load these from a file or similar.
    baseNames = {
      {TYPE_ADD, "FPAdd"},
      {TYPE_MUL, "FPMul"},
      {TYPE_LOG, "FPLog"},
      {TYPE_CATEGORICAL, "categorical"},
      {TYPE_HISTOGRAM, "histogram"},
      {TYPE_CONSTANT, "constant"}
    };

    baseNameToType = {
      {"FPAdd", TYPE_ADD},
      {"FPMul", TYPE_MUL},
      {"FPLog", TYPE_LOG},
      {"categorical", TYPE_CATEGORICAL},
      {"histogram", TYPE_HISTOGRAM},
      {"constant", TYPE_CONSTANT}
    };

    // For now these are the delays for the ufloat operators with 8;23 bits.
    // FPOps_build_mult delay: 5
    // FPOps_build_add delay: 6
    delays = {
      {TYPE_ADD, 6},
      {TYPE_MUL, 5},
      {TYPE_LOG, 0},
      {TYPE_CATEGORICAL, 1},
      {TYPE_CONSTANT, 0},
      {TYPE_HISTOGRAM, 1}
    };
  }

  void setType(Operation *op, OperatorType type) {
    //llvm::outs() << "mapping op to " << getTypeBaseName(type) << "\n";
    operatorTypes[op] = type;
  }

  OperatorType getType(Operation *op) const { return operatorTypes.at(op); }

  std::string getTypeBaseName(OperatorType type) const {
    return baseNames.at(type);
  }

  std::string getFullModuleName(Operation *op, uint64_t id) const {
    OperatorType type = getType(op);
    std::string baseName = getTypeBaseName(type);

    if (isExternalType(type))
      return baseName;

    return baseName + "_" + std::to_string(id);
  }

  bool isMapped(Operation *op) const {
    return operatorTypes.find(op) != operatorTypes.end();
  }

  uint64_t getDelay(OperatorType type) const { return delays.at(type); }
  uint64_t getDelay(std::string& baseName) const { return getDelay(baseNameToType.at(baseName)); }
};

}