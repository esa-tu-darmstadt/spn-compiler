// RUN: %optcall --spn-simplify --canonicalize --spn-type-pinning --spn-to-standard %s | FileCheck %s

module {

  "spn.joint_query"() ( {
     ^bb0(%arg0: i32): // no predecessors
      %0 = "spn.histogram"(%arg0) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}]} : (i32) -> !spn.probability
      "spn.return"(%0) : (!spn.probability) -> ()
  }) {batchSize = 1 : ui32, errorModel = 1 : i32, inputType = i32, kernelName = "spn_kernel", maxError = 2.000000e-02 : f64, numFeatures = 1 : ui32} : () -> ()

}

// Histograms get lowered to a global constant array in LLVM dialect, from which values will be loaded,
// using the input feature value as index.

// CHECK-LABEL: global_memref "private"
// CHECK-SAME: constant [[HIST:@[a-zA-Z][a-zA-Z0-9_]*]]
// CHECK-SAME: memref<2xf64>
// CHECK-SAME: dense<[2.500000e-01, 7.500000e-01]>

// CHECK-DAG: %[[#INPUT:]] = load
// CHECK-NOT: load
// CHECK-DAG: %[[#ADDRESS:]] = get_global_memref [[HIST]]
// CHECK-DAG: %[[#CAST:]] = index_cast %[[#INPUT]]
// CHECK-NOT: load
// CHECK: %[[#HIST_VAL:]] = load %[[#ADDRESS]][%[[#CAST]]]
// CHECK-NOT: load
// CHECK-NOT: store
// CHECK-DAG: %[[#LOG_VAL:]] = log %[[#HIST_VAL]]
// CHECK-NOT: store
// CHECK-DAG: store %[[#LOG_VAL]]
// CHECK-NEXT: return