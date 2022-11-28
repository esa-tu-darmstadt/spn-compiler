//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef HiSPN_INTERFACES
#define HiSPN_INTERFACES

include "mlir/IR/OpBase.td"

def LeafNodeInterface : OpInterface<"LeafNodeInterface"> {

  let description = [{
    Interface common to all SPN leaf nodes.
  }];

  let methods = [
    InterfaceMethod<"Get the feature index.", "unsigned", "getFeatureIndex">,
  ];

}

def QueryInterface : OpInterface<"QueryInterface"> {

  let description = [{
    Interface common to all SPN queries.
  }];

  let methods = [
    InterfaceMethod<"Get the number of features.", "unsigned int", "getNumFeatures">,
    InterfaceMethod<"Get the data type of the features", "mlir::Type", "getFeatureDataType">,
    InterfaceMethod<"Get the batch size to optimize for.", "unsigned int", "getBatchSize">,
    InterfaceMethod<"Get the error model for arithmetic error analysis.", "mlir::spn::high::error_model", "getErrorModel">,
    InterfaceMethod<"Get the maximum allowed arithmetic error.", "APFloat", "getMaxError">,
    InterfaceMethod<"Get the query name.", "llvm::StringRef", "getQueryName">,
  ];

}

def GraphNodeInterface : OpInterface<"GraphNodeInterface"> {

    let description = [{
        Interface common to all nodes inside the SPN DAG.
    }];

    let methods = [
        InterfaceMethod<"Get the enclosing query.", "mlir::Operation*", "getEnclosingQuery">,
    ];

}

#endif // HiSPN_INTERFACES