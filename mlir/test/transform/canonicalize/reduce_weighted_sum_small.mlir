// RUN: %optcall --canonicalize %s | FileCheck %s

module {

  "spn.joint_query"() ( {
    ^bb0(%arg0: ui32, %arg1: ui32): // no predecessors
      %2 = "spn.histogram"(%arg0) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}]} : (ui32) -> !spn.probability
      %3 = "spn.histogram"(%arg1) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}]} : (ui32) -> !spn.probability
      %4 = "spn.product"(%2, %3) : (!spn.probability, !spn.probability) -> !spn.probability
      %5 = "spn.histogram"(%arg0) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}]} : (ui32) -> !spn.probability
      %6 = "spn.histogram"(%arg1) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}]} : (ui32) -> !spn.probability
      %7 = "spn.product"(%5, %6) : (!spn.probability, !spn.probability) -> !spn.probability
      %8 = "spn.weighted_sum"(%4, %7) {weights = [1.000000e+00, 1.000000e+00]} : (!spn.probability, !spn.probability) -> !spn.probability
      %9 = "spn.weighted_sum"(%8) {weights = [7.500000e-01]} : (!spn.probability) -> !spn.probability
      "spn.return"(%9) : (!spn.probability) -> ()

  }) {batchSize = 1 : ui32, errorModel = 1 : i32, inputType = ui32, kernelName = "spn_kernel", maxError = 2.000000e-02 : f64, numFeatures = 2 : ui32} : () -> ()

}

//  This small test checks if a weighted sum is rewritten into an ordinary sum,
//  if all weights are 1.0. Furthermore, test rewrite of a weighted sum if it
//  has only one operand: into a spn.product with 2 operands (op, weight).

// CHECK-LABEL: ^bb0
// CHECK-COUNT-4: "spn.histogram"(%arg{{[0-9]+}})
// CHECK-NEXT: "spn.product"(%{{[0-9]+}}, %{{[0-9]+}})
// CHECK-NOT: "spn.weighted_sum"
// CHECK-NOT: "spn.constant"
// CHECK-NOT: "spn.product"
// CHECK-NEXT: [[ORDINARY_SUM:%[0-9]+]] = "spn.sum"(%2, %5)
// CHECK-NEXT: [[CONST_MULT:%[0-9]+]] = "spn.constant"() {value = 7.500000e-01
// CHECK-DAG: %[[#PROD:]] = "spn.product"(
// CHECK-DAG: [[ORDINARY_SUM]]
// CHECK-DAG: [[CONST_MULT]]
// CHECK-NEXT: "spn.return"(%[[#PROD]])
