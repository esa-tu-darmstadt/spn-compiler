// RUN: %optcall --canonicalize %s | FileCheck %s

module {
  "spn.single_joint"() ( {
    ^bb0(%arg0: ui32, %arg1: ui32): // no predecessors
      %0 = "spn.constant"() {value = 2.0 : f64} : () -> !spn.probability
      %1 = "spn.constant"() {value = 4.0 : f64} : () -> !spn.probability
      %2 = "spn.constant"() {value = 8.0 : f64} : () -> !spn.probability
      %3 = "spn.product"(%0, %1) : (!spn.probability, !spn.probability) -> !spn.probability
      %4 = "spn.product"(%2, %3) : (!spn.probability, !spn.probability) -> !spn.probability
      %5 = "spn.constant"() {value = 5.00000e-01 : f64} : () -> !spn.probability
      %6 = "spn.constant"() {value = 2.50000e-01 : f64} : () -> !spn.probability
      %7 = "spn.constant"() {value = 1.25000e-01 : f64} : () -> !spn.probability
      %8 = "spn.product"(%5, %6, %7) : (!spn.probability, !spn.probability, !spn.probability) -> !spn.probability
      %9 = "spn.product"(%4, %8) : (!spn.probability, !spn.probability) -> !spn.probability
      "spn.return"(%9) : (!spn.probability) -> ()
  }) {inputType = ui32, numFeatures = 2 : ui32} : () -> ()
}

//  This small test checks if a product is fold if there are constant operands.
//  In the example %9 will be constant (1.0) and can be replaced.
//  This is achieved by multiple folds & reductions, converging at const 1.0.

// CHECK-LABEL: ^bb0
// CHECK-NEXT: %[[#CONST:]] = "spn.constant"()
// CHECK-SAME: {value = 1.000000e+00 : f64}
// CHECK-NEXT: "spn.return"(%[[#CONST]])
