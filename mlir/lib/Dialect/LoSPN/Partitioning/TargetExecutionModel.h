//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================
#pragma once

#include <string>

namespace mlir {
class Operation;
class Value;
namespace spn::low::partitioning {

class TargetExecutionModel {
public:
  virtual ~TargetExecutionModel() = default;

  virtual std::string getTargetName() const { return "Generic"; }

  /// Returns the number of processors.
  virtual int getNumProcessors() const { return 1; }

  /// Returns the number of threads per processor.
  virtual int getNumThreadsPerProcessor() const { return 1; }

  /// Returns the cost to compute the given operation.
  virtual int getCostOfComputation(mlir::Operation *op) const { return 1; }

  /// Returns the cost to communicate the given value between two processors.
  virtual int getCostOfInterProcCommunication(mlir::Value &value) const { return 1; }

  /// Returns the cost to communicate the given value between two threads within the same processor.
  virtual int getCostOfIntraProcCommunication(mlir::Value &value) const { return 0; }
};

TargetExecutionModel &getGenericTargetExecutionModel();
} // namespace spn::low::partitioning
} // namespace mlir