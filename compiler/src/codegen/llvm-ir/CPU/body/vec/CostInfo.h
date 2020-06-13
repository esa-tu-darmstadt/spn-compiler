#pragma once
#include "llvm/Analysis/TargetTransformInfo.h"

#define AVX2 1

#define ARCH AVX2
class CostInfo {
 public:
  CostInfo(size_t width);
  size_t histogramCost(std::multiset<size_t, std::greater<size_t>> inputHistos);
  size_t gaussCost(std::multiset<size_t> inputGaussians);
  size_t getHistogramPenalty(size_t width);
  float getExtractCost(size_t width);
  float getInsertCost(size_t width);
  size_t vecArithCost;
  size_t insertCost;
  size_t extractCost;
  size_t scalarArithCost;
  size_t gaussArithCost;
};
