//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "LoSPN/Bufferize/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::spn::low;

namespace mlir::spn::low {

#define GEN_PASS_DEF_LOSPNBUFFERIZE
#include "LoSPN/LoSPNPasses.h.inc"

struct LoSPNBufferize : public impl::LoSPNBufferizeBase<LoSPNBufferize> {
protected:
  void runOnOperation() override {
    bufferization::OneShotBufferizationOptions options;
    if (failed(bufferization::bufferizeOp(getOperation(), options)))
      signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    impl::LoSPNBufferizeBase<LoSPNBufferize>::getDependentDialects(registry);
    ::mlir::spn::low::registerBufferizableOpInterfaceExternalModels(registry);
  }
};

} // namespace mlir::spn::low