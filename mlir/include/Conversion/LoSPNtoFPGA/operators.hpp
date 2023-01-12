#pragma once

#include <unordered_map>

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "LoSPN/LoSPNDialect.h"
#include "LoSPN/LoSPNOps.h"

#include "llvm/ADT/TypeSwitch.h"


namespace mlir::spn::fpga::operators {

using namespace ::mlir::spn::low;

enum OperatorType {
  // these come from external verilog sources
  TYPE_ADD, TYPE_MUL, TYPE_LOG,
  // these are generated
  TYPE_CATEGORICAL, TYPE_HISTOGRAM, TYPE_CONSTANT,
  // not important for scheduling
  TYPE_YIELD, TYPE_BODY
};

inline const std::array<OperatorType, 8> OPERATOR_TYPES = {
  TYPE_ADD, TYPE_MUL, TYPE_LOG,
  TYPE_CATEGORICAL, TYPE_HISTOGRAM, TYPE_CONSTANT,
  TYPE_YIELD, TYPE_BODY
};

enum OperatorPortType {
  PORT_SIGNAL, PORT_INDEX, PORT_PROBABILITY
};

enum OperatorPortDirection {
  INPUT, OUTPUT
};

struct OperatorPortInfo {
  std::string name;
  OperatorPortDirection direction;
  OperatorPortType type;
};

// Since everything becomes a InstanceOp or similar primitive HW type it
// is useful to track the original operator types.
class OperatorTypeMapping {
  std::unordered_map<Operation *, OperatorType> operatorTypes;
  std::unordered_map<OperatorType, std::string> baseNames;
  std::unordered_map<std::string, OperatorType> baseNameToType;
  std::unordered_map<OperatorType, uint64_t> delays;

  static bool isExternalType(OperatorType type) {
    return type == TYPE_ADD || type == TYPE_MUL || type == TYPE_LOG;
  }

  static OperatorPortInfo inPort(const std::string& name, OperatorPortType type) {
    return OperatorPortInfo{ .name = name, .direction = INPUT, .type = type };
  }

  static OperatorPortInfo outPort(const std::string& name, OperatorPortType type) {
    return OperatorPortInfo{ .name = name, .direction = OUTPUT, .type = type };
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

  void setType(Operation *op, OperatorType type) { operatorTypes[op] = type; }
  OperatorType getType(Operation *op) const { return operatorTypes.at(op); }
  std::string getTypeBaseName(OperatorType type) const { return baseNames.at(type); }

  std::string getFullModuleName(Operation *op, uint64_t id) const {
    OperatorType type = getType(op);
    std::string baseName = getTypeBaseName(type);

    if (isExternalType(type))
      return baseName;

    return baseName + "_" + std::to_string(id);
  }

  std::string getFullModuleName(OperatorType type, uint64_t id) const {
    std::string baseName = getTypeBaseName(type);

    if (isExternalType(type))
      return baseName;

    return baseName + "_" + std::to_string(id);
  }

  bool isMapped(Operation *op) const { return operatorTypes.find(op) != operatorTypes.end(); }
  uint64_t getDelay(OperatorType type) const { return delays.at(type); }
  uint64_t getDelay(std::string& baseName) const { return getDelay(baseNameToType.at(baseName)); }

  virtual std::vector<OperatorPortInfo> getOperatorPorts(OperatorType type) const {
    switch (type) {
      case TYPE_ADD: case TYPE_MUL:
        return std::vector<OperatorPortInfo>{
          inPort("clock", PORT_SIGNAL),
          inPort("reset", PORT_SIGNAL),
          inPort("io_a", PORT_PROBABILITY),
          inPort("io_b", PORT_PROBABILITY),
          outPort("io_r", PORT_PROBABILITY)
        };
      case TYPE_CATEGORICAL: case TYPE_HISTOGRAM:
        return std::vector<OperatorPortInfo>{
          inPort("clk", PORT_SIGNAL),
          inPort("rst", PORT_SIGNAL),
          inPort("in_index", PORT_INDEX),
          outPort("out_prob", PORT_PROBABILITY)
        };
      case TYPE_CONSTANT:
        return std::vector<OperatorPortInfo>{
          inPort("clk", PORT_SIGNAL),
          inPort("rst", PORT_SIGNAL),
          outPort("out_const", PORT_PROBABILITY)
        };
      case TYPE_LOG:
        return std::vector<OperatorPortInfo>{
          inPort("clk", PORT_SIGNAL),
          inPort("rst", PORT_SIGNAL),
          inPort("in_a", PORT_PROBABILITY),
          outPort("out_b", PORT_PROBABILITY)
        };
      case TYPE_BODY:
        return std::vector<OperatorPortInfo>{
          inPort("clk", PORT_SIGNAL),
          inPort("rst", PORT_SIGNAL),
          outPort("out_prob", PORT_PROBABILITY)
        };
      default:
        assert(false);
    }
  }

  static OperatorType getOperatorType(Operation *op) {
    OperatorType opType = llvm::TypeSwitch<Operation *, OperatorType>(op) 
      .Case<SPNAdd>([&](SPNAdd op) { return TYPE_ADD; })
      .Case<SPNMul>([&](SPNMul op) { return TYPE_MUL; })
      .Case<SPNCategoricalLeaf>([&](SPNCategoricalLeaf op) { return TYPE_CATEGORICAL; })
      .Case<SPNConstant>([&](SPNConstant op) { return TYPE_CONSTANT; })
      .Case<SPNLog>([&](SPNLog op) { return TYPE_LOG; })
      .Case<SPNHistogramLeaf>([&](SPNHistogramLeaf op) { return TYPE_HISTOGRAM; })
      .Case<SPNYield>([&](SPNYield op) { return TYPE_YIELD; })
      .Default([&](Operation *op) -> OperatorType {
        assert(false && "Unexpected type");
      });

    return opType;
  }
};

}