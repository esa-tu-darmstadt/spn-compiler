//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_KERNEL_H
#define SPNC_KERNEL_H

#include <optional>
#include <cstdlib>
#include <cstdint>
#include <variant>
#include <string>
#include <sstream>

///
/// Namespace for all entities related to the SPN compiler.
///
namespace spnc {

  enum KernelQueryType : unsigned { JOINT_QUERY = 1 };

  enum KernelTarget : unsigned { CPU = 1, CUDA = 2, FPGA = 3 };

  enum KernelType : unsigned { CLASSICAL_KERNEL = 1, FPGA_KERNEL = 2 };

  ///
  /// Represents a kernel that is generated by the compiler and can be loaded and executed by the runtime.
  /// Contains information about the file-path of the generated kernel and the name of the generated
  /// function inside that file.
  ///
  struct ClassicalKernel {
    std::string fileName;
    std::string kernelName;
    unsigned query;
    unsigned targetArch;
    unsigned batchSize;
    unsigned numFeatures;
    unsigned bytesPerFeature;
    unsigned numResults;
    unsigned bytesPerResult;
    std::string dtype;

    size_t uniqueId() const {
      return std::hash<std::string>{}(fileName + kernelName);
    }
  };

  // The fields in this struct are alloweded to contain only partially valid information. This depends on the use case.
  struct FPGAKernel {
    std::string fileName = "N/A";
    std::string kernelName = "N/A";
    int32_t kernelId = -1;
    std::string deviceName = "N/A";
    int32_t deviceSpeed = -1;

    int32_t bodyDelay = -1;
    int32_t fifoDepth = -1;
    // these can be weird numbers like 31 bits
    int32_t spnVarCount = -1;
    int32_t spnBitsPerVar = -1;
    int32_t spnResultWidth = -1;

    // sets the width for S_AXIS_CONTROLLER and M_AXIS_CONTROLLER
    // + sets the widths of the SPNController input/output AXIStreams
    int32_t mAxisControllerWidth = -1;
    int32_t sAxisControllerWidth = -1;

    // sets the width for S_AXIS and M_AXIS and also M_AXI
    int32_t memDataWidth = -1;
    int32_t memAddrWidth = -1;

    // sets the width for S_AXI_LITE
    int32_t liteDataWidth = -1;
    int32_t liteAddrWidth = -1;

    std::string to_string() const {
      return (std::stringstream{}
        << "FPGAKernel{"
        << "fileName=" << fileName
        << ", kernelName=" << kernelName
        << ", kernelId=" << kernelId
        << ", bodyDelay=" << bodyDelay
        << ", fifoDepth=" << fifoDepth
        << ", spnVarCount=" << spnVarCount
        << ", spnBitsPerVar=" << spnBitsPerVar
        << ", spnResultWidth=" << spnResultWidth
        << ", mAxisControllerWidth=" << mAxisControllerWidth
        << ", sAxisControllerWidth=" << sAxisControllerWidth
        << ", memDataWidth=" << memDataWidth
        << ", memAddrWidth=" << memAddrWidth
        << ", liteDataWidth=" << liteDataWidth
        << ", liteAddrWidth=" << liteAddrWidth
        << "}"
      ).str();
    }
  };

  class Kernel {
  public:
    std::variant<ClassicalKernel, FPGAKernel> kernel;
  public:
    Kernel() {}

    Kernel(const ClassicalKernel& kernel):
      kernel(kernel) {}
    
    Kernel(const FPGAKernel& kernel):
      kernel(kernel) {}

    KernelType getKernelType() const {
      return KernelType(kernel.index() + 1);
    }

    const ClassicalKernel& getClassicalKernel() const {
      return std::get<ClassicalKernel>(kernel);
    }

    ClassicalKernel& getClassicalKernel() {
      return std::get<ClassicalKernel>(kernel);
    }

    const FPGAKernel& getFPGAKernel() const {
      return std::get<FPGAKernel>(kernel);
    }

    FPGAKernel& getFPGAKernel() {
      return std::get<FPGAKernel>(kernel);
    }
  };
}

namespace spnc {

template <class T>
T roundN(const T& n, const T& N) {
  if (n % N == 0)
    return n;
  return n + (N - n % N);
}

}

#endif //SPNC_KERNEL_H
