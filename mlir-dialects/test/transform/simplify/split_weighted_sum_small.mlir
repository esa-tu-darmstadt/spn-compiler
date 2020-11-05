// RUN: %optcall --spn-simplify %s | FileCheck %s

module {
  "spn.single_joint"() ( {
    ^bb0(%arg0: ui32, %arg1: ui32): // no predecessors
      %2 = "spn.histogram"(%arg0) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}]} : (ui32) -> !spn.probability
      %3 = "spn.histogram"(%arg1) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}]} : (ui32) -> !spn.probability
      %4 = "spn.product"(%2, %3) : (!spn.probability, !spn.probability) -> !spn.probability
      %5 = "spn.histogram"(%arg0) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}]} : (ui32) -> !spn.probability
      %6 = "spn.histogram"(%arg1) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}]} : (ui32) -> !spn.probability
      %7 = "spn.product"(%5, %6) : (!spn.probability, !spn.probability) -> !spn.probability
      %8 = "spn.weighted_sum"(%4, %7) {weights = [3.000000e-01, 0.69999999999999996]} : (!spn.probability, !spn.probability) -> !spn.probability
      "spn.return"(%8) : (!spn.probability) -> ()
  }) {inputType = ui32, kernelName = "spn_kernel", numFeatures = 2 : ui32} : () -> ()
}

//  This small test checks if a weighted sum is removed and replaced by pairs of
//  <constant, product> (one for each weight and operand) and one ordinary sum.

// CHECK-LABEL: ^bb0
// CHECK-COUNT-4: "spn.histogram"(%arg{{[0-9]+}})
// CHECK-NEXT: "spn.product"(%{{[0-9]+}}, %{{[0-9]+}})
// CHECK-NOT: "spn.weighted_sum"
// CHECK-COUNT-2: {{[[:blank:]]%[0-9]+ = "spn.constant"\(\) [[:graph:][:blank:]]+([[:space:]]{1})[[:blank:]]*%[0-9]+ = "spn.product"\(%[0-9]+, %[0-9]+\) [[:graph:][:blank:]]+}}
// CHECK-NEXT: "spn.sum"(%{{[0-9]+}}, %{{[0-9]+}})
// CHECK-NEXT: "spn.return"
