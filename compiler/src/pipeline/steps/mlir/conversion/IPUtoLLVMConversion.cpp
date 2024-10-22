//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "IPUtoLLVMConversion.h"

using namespace mlir;

void spnc::IPUtoLLVMConversion::initializePassPipeline(mlir::PassManager *pm,
                                                       mlir::MLIRContext *ctx) {
  /**
    Input here:
    module {
      func.func @task_0(%arg0: memref<?x10xf64>, %arg1: memref<1x?xf32>) {

        }
        return
      }
      func.func @spn_kernel(%arg0: memref<?x10xf64>, %arg1: memref<1x?xf32>) {
        call @task_0(%arg0, %arg1) : (memref<?x10xf64>, memref<1x?xf32>) -> ()
        return
      }
    }

    1. Turn task into codelet
    2. Turn kernel into graph + program
   */
}