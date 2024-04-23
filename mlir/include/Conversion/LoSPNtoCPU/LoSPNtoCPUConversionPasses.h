//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_LOSPNTOCPUCONVERSIONPASSES_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_LOSPNTOCPUCONVERSIONPASSES_H

#include "LoSPNtoCPU/Vectorization/SLP/Util.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir {
namespace spn {

struct LoSPNtoCPUStructureConversionPass
    : public PassWrapper<LoSPNtoCPUStructureConversionPass,
                         OperationPass<ModuleOp>> {
public:
  LoSPNtoCPUStructureConversionPass() = default;
  /// Constructor for accepting arguments from the driver instead of spnc-opt.
  LoSPNtoCPUStructureConversionPass(bool vectorize, unsigned maxAttempts,
                                    unsigned maxSuccessfulIterations,
                                    unsigned maxNodeSize, unsigned maxLookAhead,
                                    bool reorderInstructionsDFS,
                                    bool allowDuplicateElements,
                                    bool allowTopologicalMixing,
                                    bool useXorChains) {
    this->vectorize.setValue(vectorize);
    this->maxAttempts.setValue(maxAttempts);
    this->maxSuccessfulIterations.setValue(maxSuccessfulIterations);
    this->maxNodeSize.setValue(maxNodeSize);
    this->maxLookAhead.setValue(maxLookAhead);
    this->reorderInstructionsDFS.setValue(reorderInstructionsDFS);
    this->allowDuplicateElements.setValue(allowDuplicateElements);
    this->allowTopologicalMixing.setValue(allowTopologicalMixing);
    this->useXorChains.setValue(useXorChains);
  }
  LoSPNtoCPUStructureConversionPass(
      LoSPNtoCPUStructureConversionPass const &pass)
      : PassWrapper<LoSPNtoCPUStructureConversionPass, OperationPass<ModuleOp>>(
            pass) {}

  void getDependentDialects(DialectRegistry &registry) const override;

  StringRef getArgument() const override {
    return "convert-lospn-structure-to-cpu";
  }
  StringRef getDescription() const override {
    return "Convert structure from LoSPN to CPU target";
  }

  Option<bool> vectorize{
      *this, "cpu-vectorize",
      llvm::cl::desc("Vectorize code generated for CPU targets"),
      llvm::cl::init(false)};
  Option<unsigned> maxAttempts{
      *this, "slp-max-attempts",
      llvm::cl::desc("Maximum number of SLP vectorization attempts"),
      llvm::cl::init(1)};
  Option<unsigned> maxSuccessfulIterations{
      *this, "slp-max-successful-iterations",
      llvm::cl::desc("Maximum number of successful SLP vectorization runs to "
                     "be applied to a function"),
      llvm::cl::init(1)};
  Option<unsigned> maxNodeSize{
      *this, "slp-max-node-size",
      llvm::cl::desc("Maximum multinode size during SLP vectorization in terms "
                     "of the number of vectors they may contain"),
      llvm::cl::init(10)};
  Option<unsigned> maxLookAhead{
      *this, "slp-max-look-ahead",
      llvm::cl::desc("Maximum look-ahead depth when reordering multinode "
                     "operands during SLP vectorization"),
      llvm::cl::init(3)};
  Option<bool> reorderInstructionsDFS{
      *this, "slp-reorder-instructions-dfs",
      llvm::cl::desc("Flag to indicate if SLP-vectorized instructions should "
                     "be arranged in DFS order (true) or in BFS order (false)"),
      llvm::cl::init(true)};
  Option<bool> allowDuplicateElements{
      *this, "slp-allow-duplicate-elements",
      llvm::cl::desc("Flag to indicate whether duplicate elements are allowed "
                     "in vectors during SLP graph building"),
      llvm::cl::init(false)};
  Option<bool> allowTopologicalMixing{
      *this, "slp-allow-topological-mixing",
      llvm::cl::desc("Flag to indicate if elements with different topological "
                     "depths are allowed in vectors during SLP graph building"),
      llvm::cl::init(false)};
  Option<bool> useXorChains{
      *this, "slp-use-xor-chains",
      llvm::cl::desc("Flag to indicate if XOR chains should be used to compute "
                     "look-ahead scores instead of Porpodas's algorithm"),
      llvm::cl::init(true)};

  StringRef getArgument() const override {
    return "convert-lospn-structure-to-cpu";
  }
  StringRef getDescription() const override {
    return "Convert structure from LoSPN to CPU target";
  }

protected:
  void runOnOperation() override;
};

struct LoSPNtoCPUNodeConversionPass
    : public PassWrapper<LoSPNtoCPUNodeConversionPass,
                         OperationPass<ModuleOp>> {

public:
  void getDependentDialects(DialectRegistry &registry) const override;
  StringRef getArgument() const override {
    return "convert-lospn-nodes-to-cpu";
  }
  StringRef getDescription() const override {
    return "Convert nodes from LoSPN to CPU target";
  }

  StringRef getArgument() const override {
    return "convert-lospn-nodes-to-cpu";
  }
  StringRef getDescription() const override {
    return "Convert nodes from LoSPN to CPU target";
  }

protected:
  void runOnOperation() override;
};

std::unique_ptr<Pass> createLoSPNtoCPUNodeConversionPass();

struct LoSPNNodeVectorizationPass
    : public PassWrapper<LoSPNNodeVectorizationPass, OperationPass<ModuleOp>> {

public:
  void getDependentDialects(DialectRegistry &registry) const override;

  StringRef getArgument() const override { return "vectorize-lospn-nodes"; }
  StringRef getDescription() const override {
    return "Vectorize LoSPN nodes for CPU target";
  }

protected:
  void runOnOperation() override;
  StringRef getArgument() const override { return "vectorize-lospn-nodes"; }
  StringRef getDescription() const override {
    return "Vectorize LoSPN nodes for CPU target";
  }
};

std::unique_ptr<Pass> createLoSPNNodeVectorizationPass();

} // namespace spn
} // namespace mlir

#endif // SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_LOSPNTOCPUCONVERSIONPASSES_H
