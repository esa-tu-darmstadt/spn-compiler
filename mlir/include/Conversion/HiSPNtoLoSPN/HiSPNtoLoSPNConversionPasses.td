//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef MLIR_CONVERSION_HISPNTOLOSPNCONVERSIONPASSES
#define MLIR_CONVERSION_HISPNTOLOSPNCONVERSIONPASSES

include "mlir/Pass/PassBase.td"

def HiSPNtoLoSPNNodeConversionPass : Pass<"convert-hispn-node-to-lospn", "ModuleOp"> {
    let summary = "Convert HiSPN nodes to LoSPN nodes";

    let dependentDialects = [
        "mlir::spn::low::LoSPNDialect"
    ];

    let options = [
        Option<"computeLogSpace", "use-log-space", "bool", "false", "Enable log-space computation for numerical stability">,
        Option<"optimizeRepresentation", "opt-repr", "bool", "false", "Whether an optimal representation for SPN evaluation should be determined.">,
        Option<"computeTypeWidth", "compute-type-width", "int", "64", "The width of the floating-point compute type.">,
        Option<"logComputeTypeWidth", "log-compute-type-width", "int", "32", "The width of the floating-point logarithmic compute type.">,
    ];
}

def HiSPNtoLoSPNQueryConversionPass : Pass<"convert-hispn-query-to-lospn", "ModuleOp"> {
    let summary = "Convert queries from HiSPN to LoSPN dialect";

    let dependentDialects = [
        "mlir::spn::low::LoSPNDialect"
    ];

    let options = [
        Option<"computeLogSpace", "use-log-space", "bool", "false", "Enable log-space computation for numerical stability">,
        Option<"optimizeRepresentation", "opt-repr", "bool", "false", "Whether an optimal representation for SPN evaluation should be determined.">,
        Option<"computeTypeWidth", "compute-type-width", "int", "64", "The width of the floating-point compute type.">,
        Option<"logComputeTypeWidth", "log-compute-type-width", "int", "32", "The width of the floating-point logarithmic compute type.">,
    ];
}


#endif // MLIR_CONVERSION_HISPNTOLOSPNCONVERSIONPASSES