#pragma once
#include "llvm/Analysis/TargetTransformInfo.h"

#define AVX2 1

#define ARCH AVX2
class CostInfo {
 public:
  CostInfo(size_t width);
  size_t vecArithCost;
  size_t insertCost;
  size_t extractCost;
  size_t scalarArithCost;
};
