// RUN: %optcall --spn-simplify --canonicalize --spn-type-pinning --spn-to-standard --spn-to-llvm %s | FileCheck %s

module {

  "spn.single_joint"() ( {
     ^bb0(%arg0: ui32): // no predecessors
      %0 = "spn.histogram"(%arg0) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}]} : (ui32) -> !spn.probability
      "spn.return"(%0) : (!spn.probability) -> ()
  }) {inputType = ui32, numFeatures = 1 : ui32} : () -> ()

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
// CHECK-DAG: llvm.store %[[#HIST_VAL]]
// CHECK-NEXT: llvm.return