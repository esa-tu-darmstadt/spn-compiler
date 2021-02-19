//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "TargetInformation.h"
#include "llvm/Support/Host.h"
#include "llvm/ADT/StringRef.h"
#include <llvm/ADT/StringMap.h>
#include <iostream>
#include <sstream>
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_os_ostream.h"
#include "mlir/IR/BuiltinTypes.h"

mlir::spn::TargetInformation::TargetInformation() {
  llvm::dbgs() << "Host CPU name:" << llvm::sys::getHostCPUName().str() << "\n";
  llvm::dbgs() << "Host process triple: " << llvm::sys::getProcessTriple() << "\n";
  llvm::dbgs() << "Host default target triple: " << llvm::sys::getDefaultTargetTriple() << "\n";
  llvm::dbgs() << "Host physical core count: " << llvm::sys::getHostNumPhysicalCores() << "\n";
  llvm::sys::getHostCPUFeatures(featureMap);
  std::stringstream featureList;
  bool initial = true;
  for (auto& s : featureMap.keys()) {
    if (featureMap.lookup(s)) {
      if (!initial) {
        featureList << ", ";
      }
      featureList << s.str();
      initial = false;
    }
  }
  llvm::dbgs() << "Host CPU feature list: " << featureList.str() << "\n";
}

mlir::spn::TargetInformation& mlir::spn::TargetInformation::nativeCPUTarget() {
  static TargetInformation ti;
  return ti;
}

bool mlir::spn::TargetInformation::hasAVX2Support() {
  return featureMap.lookup("avx2");
}

bool mlir::spn::TargetInformation::hasAVX512Support() {
  return featureMap.lookup("avx512f");
}

bool mlir::spn::TargetInformation::hasAVXSupport() {
  return featureMap.lookup("avx");
}

unsigned int mlir::spn::TargetInformation::getHWVectorEntries(mlir::Type type) {
  if (hasAVX512Support()) {
    return getHWVectorEntriesAVX512(type);
  }
  if (hasAVX2Support()) {
    return getHWVectorEntriesAVX2(type);
  }
  if (hasAVXSupport()) {
    return getHWVectorEntriesAVX(type);
  }
  return 1;
}

unsigned int mlir::spn::TargetInformation::getHWVectorEntriesAVX2(mlir::Type type) {
  if (auto intType = type.dyn_cast<IntegerType>()) {
    // AVX2 extends most integer instructions to 256 bit.
    return 256 / intType.getWidth();
  }
  if (auto floatType = type.dyn_cast<FloatType>()) {
    switch (floatType.getWidth()) {
      case 64: return 4;
      case 32: return 8;
        // Float16 can only be used for store/load but not for arithmetic on most AVX2 implementations.
      default: return 8;
    }
  }
  return 1;
}

unsigned int mlir::spn::TargetInformation::getHWVectorEntriesAVX(mlir::Type type) {
  if (auto intType = type.dyn_cast<IntegerType>()) {
    // With AVX, integer instructions are only defined for 128 bit.
    return 128 / intType.getWidth();
  }
  if (auto floatType = type.dyn_cast<FloatType>()) {
    switch (floatType.getWidth()) {
      case 64: return 4;
      case 32: return 8;
        // Float16 can only be used for store/load but not for arithmetic on most AVX2 implementations.
      default: return 8;
    }
  }
  return 1;
}

unsigned int mlir::spn::TargetInformation::getHWVectorEntriesAVX512(mlir::Type type) {
  if (auto intType = type.dyn_cast<IntegerType>()) {
    // AVX2 extends most integer instructions to 256 bit.
    return 512 / intType.getWidth();
  }
  if (auto floatType = type.dyn_cast<FloatType>()) {
    switch (floatType.getWidth()) {
      case 64: return 8;
      case 32: return 16;
        // Float16 can only be used for store/load but not for arithmetic on most AVX2 implementations.
      default: return 16;
    }
  }
  return 1;
}

