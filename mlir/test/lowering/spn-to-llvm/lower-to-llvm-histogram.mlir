// RUN: %optcall --spn-simplify --canonicalize --spn-type-pinning --spn-to-standard --spn-to-llvm %s | FileCheck %s

module {

  "spn.joint_query"() ( {
     ^bb0(%arg0: i32): // no predecessors
      %0 = "spn.histogram"(%arg0) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}]} : (i32) -> !spn.probability
      "spn.return"(%0) : (!spn.probability) -> ()
  }) {batchSize = 1 : ui32, errorModel = 1 : i32, inputType = i32, kernelName = "spn_kernel", maxError = 2.000000e-02 : f64, numFeatures = 1 : ui32} : () -> ()

}

// Histograms get lowered to a global constant array in LLVM dialect, from which values will be loaded,
// using the input feature value as index.

// CHECK-LABEL: llvm.mlir.global
// CHECK-SAME: constant [[HIST:@[a-zA-Z][a-zA-Z0-9_]*]]
// CHECK-SAME: dense<[2.500000e-01, 7.500000e-01]> : tensor<2xf64>
// CHECK-SAME: !llvm.array<2 x double>

// CHECK-DAG: %[[#INPUT:]] = llvm.load
// CHECK-NOT: llvm.load
// CHECK-DAG: %[[#ADDRESS:]] = llvm.mlir.addressof [[HIST]]
// CHECK-DAG: %[[#CONST0:]] = llvm.mlir.constant(0 : i64)
// CHECK-NOT: llvm.load
// CHECK: %[[#GEP:]] = llvm.getelementptr %[[#ADDRESS]][%[[#CONST0]], %[[#INPUT]]]
// CHECK-NEXT: %[[#HIST_VAL:]] = llvm.load %[[#GEP]]
// CHECK-NOT: llvm.load
// CHECK-NOT: llvm.store
// CHECK-DAG: %[[#LOG_VAL:]] = "llvm.intr.log"(%[[#HIST_VAL]])
// CHECK-NOT: llvm.store
// CHECK-DAG: llvm.store %[[#LOG_VAL]]
// CHECK-NEXT: llvm.return