// RUN: %optcall --canonicalize %s | FileCheck %s

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
      %9 = "spn.constant"() {value = 7.0 : f64} : () -> !spn.probability
      %10 = "spn.constant"() {value = 23.0 : f64} : () -> !spn.probability
      %11 = "spn.constant"() {value = 42.0 : f64} : () -> !spn.probability
      %12 = "spn.product"(%9, %10) : (!spn.probability, !spn.probability) -> !spn.probability
      %13 = "spn.product"(%10, %11) : (!spn.probability, !spn.probability) -> !spn.probability
      %14 = "spn.weighted_sum"(%8, %12, %13) {weights = [1.250000e-01, 2.500000e-01, 5.000000e-01]} : (!spn.probability, !spn.probability, !spn.probability) -> !spn.probability
      "spn.return"(%14) : (!spn.probability) -> ()
  }) {inputType = ui32, kernelName = "spn_kernel", numFeatures = 2 : ui32} : () -> ()

}

//  This small test checks if a weighted sum is fold if there are constant
//  operands. In the example %7 is dependent of external parameters (non-const).
//  But %9, %10, %11 are const, which is why %12, %13 are also const.
//  Therefore %14 can be fold into a 2-opCount weighted sum.
//  Note: 5.232500e+02 = ( 0.25 * [7 * 23]  +  0.5 * [23 * 42] )

// CHECK-LABEL: ^bb0
// CHECK-COUNT-4: "spn.histogram"(%arg{{[0-9]+}})
// CHECK-NEXT: "spn.product"(%{{[0-9]+}}, %{{[0-9]+}}) {{[[:graph:][:blank:]]+}}
// CHECK-NEXT: [[NON_CONST_ADDEND:%[0-9]+]] = "spn.weighted_sum"
// CHECK-NEXT: [[CONST_ADDEND:%[0-9]+]] = "spn.constant"()
// CHECK-SAME: value = 5.232500e+02 : f64
// CHECK-NOT: "spn.constant"
// CHECK-NOT: "spn.sum"
// CHECK-DAG: %[[#SUM:]] = "spn.weighted_sum"(
// CHECK-DAG: [[NON_CONST_ADDEND]]
// CHECK-DAG: [[CONST_ADDEND]]
// CHECK-NEXT: "spn.return"(%[[#SUM]])
