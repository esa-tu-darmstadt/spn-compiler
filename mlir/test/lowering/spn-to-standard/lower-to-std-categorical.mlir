// RUN: %optcall --spn-simplify --canonicalize --spn-type-pinning --spn-to-standard %s | FileCheck %s

module {
  "spn.joint_query"() ( {
  ^bb0(%arg0: i8): // no predecessors
    %0 = "spn.categorical"(%arg0) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01]} : (i8) -> !spn.probability
    "spn.return"(%0) : (!spn.probability) -> ()
  }) {batchSize = 1 : ui32, errorModel = 1 : i32, inputType = i8, kernelName = "test", maxError = 2.000000e-02 : f64, numFeatures = 1 :ui32} : () -> ()
}

// Categorical distributions get lowered to a global constant array in LLVM dialect, from which values will be loaded,
// using the input feature value as index.

// CHECK-LABEL: global_memref "private"
// CHECK-SAME: constant [[CATEGORICAL:@[a-zA-Z][a-zA-Z0-9_]*]]
// CHECK-SAME: memref<3xf64>
// CHECK-SAME: dense<[3.500000e-01, 5.500000e-01, 1.000000e-01]>

// CHECK-DAG: %[[#INPUT:]] = load
// CHECK-NOT: load
// CHECK-DAG: %[[#ADDRESS:]] = get_global_memref [[CATEGORICAL]]
// CHECK-DAG: %[[#CAST:]] = index_cast %[[#INPUT]]
// CHECK-NOT: load
// CHECK: %[[#CAT_VAL:]] = load %[[#ADDRESS]][%[[#CAST]]]
// CHECK-NOT: load
// CHECK-NOT: store
// CHECK-DAG: %[[#LOG_VAL:]] = log %[[#CAT_VAL]]
// CHECK-NOT: store
// CHECK-DAG: store %[[#LOG_VAL]]
// CHECK-NEXT: return
