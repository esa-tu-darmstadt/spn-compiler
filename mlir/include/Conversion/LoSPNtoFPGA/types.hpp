#pragma once

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"


namespace mlir::spn::fpga::types {

class TargetTypes {
  OpBuilder& builder;
  Type indexType, probType, signalType;
public:
  TargetTypes(OpBuilder& builder): builder(builder) {
    indexType = builder.getI8Type();
    probType = builder.getIntegerType(31);
    signalType = builder.getI1Type();
  }

  Type getIndexType() const { return indexType; }
  Type getProbType() const { return probType; }
  Type getSignalType() const { return signalType; }

  uint64_t convertProb(double prob) const {
    float f32 = float(prob);
    // remove the sign bit
    return *reinterpret_cast<const uint32_t *>(&f32) & 0x7fffffff;
  }
};

}