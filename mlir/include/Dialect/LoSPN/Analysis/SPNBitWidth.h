#pragma once

#include "LoSPN/LoSPNOps.h"


namespace mlir::spn {

class SPNBitWidth {
  uint32_t bitsPerVar = 0;

  void analyzeGraph(Operation* root);
  void updateBits(uint32_t entryCount);
public:
  explicit SPNBitWidth(Operation* root);
  uint32_t getBitsPerVar() const { return bitsPerVar; }
};

}