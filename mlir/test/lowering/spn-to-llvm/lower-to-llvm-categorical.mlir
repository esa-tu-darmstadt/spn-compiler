// RUN: %optcall --spn-simplify --canonicalize --spn-type-pinning --spn-to-standard --spn-to-llvm %s | FileCheck %s

module {
  "spn.joint_query"() ( {
  ^bb0(%arg0: i8): // no predecessors
    %0 = "spn.categorical"(%arg0) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01]} : (i8) -> !spn.probability
    "spn.return"(%0) : (!spn.probability) -> ()
  }) {batchSize = 1 : ui32, errorModel = 1 : i32, inputType = i8, kernelName = "test", maxError = 2.000000e-02 : f64, numFeatures = 1 :ui32} : () -> ()
}

// CHECK-LABEL: llvm.mlir.global
// CHECK-SAME: constant [[CATEGORICAL:@[a-zA-Z][a-zA-Z0-9_]*]]
// CHECK-SAME: dense<[3.500000e-01, 5.500000e-01, 1.000000e-01]> : tensor<3xf64>
// CHECK-SAME: !llvm.array<3 x double>

// CHECK-DAG: %[[#INPUT:]] = llvm.load
// CHECK-NOT: llvm.load
// CHECK-DAG: %[[#ADDRESS:]] = llvm.mlir.addressof [[CATEGORICAL]]
// CHECK-DAG: %[[#CONST0:]] = llvm.mlir.constant(0 : i64)
// CHECK-NOT: llvm.load
// CHECK: %[[#GEP:]] = llvm.getelementptr %[[#ADDRESS]][%[[#CONST0]], %[[#INPUT]]]
// CHECK-NEXT: %[[#CAT_VAL:]] = llvm.load %[[#GEP]]
// CHECK-NOT: llvm.load
// CHECK-NOT: llvm.store
// CHECK-DAG: %[[#LOG_VAL:]] = "llvm.intr.log"(%[[#CAT_VAL]])
// CHECK-NOT: llvm.store
// CHECK-DAG: llvm.store %[[#LOG_VAL]]
// CHECK-NEXT: llvm.return
