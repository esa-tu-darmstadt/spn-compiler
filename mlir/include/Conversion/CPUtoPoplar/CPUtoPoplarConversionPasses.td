//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef MLIR_CONVERSION_CPUTOPOPLARCONVERSIONPASSES
#define MLIR_CONVERSION_CPUTOPOPLARCONVERSIONPASSES

include "mlir/Pass/PassBase.td"

def CPUtoPoplarConversionPass : Pass<"convert-cpu-to-poplar", "ModuleOp"> {
  let summary = "Convert kernel and task functions from the func dialect to the poplar dialect";
  let dependentDialects = ["::mlir::ipu::poplar::PoplarDialect", "::mlir::spn::low::LoSPNDialect", "::mlir::func::FuncDialect"];
}

#endif // MLIR_CONVERSION_CPUTOPOPLARCONVERSIONPASSES