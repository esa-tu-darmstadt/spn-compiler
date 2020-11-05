// RUN: %optcall --canonicalize %s | FileCheck %s

module {

  "spn.single_joint"() ( {
    ^bb0(%arg0: ui32, %arg1: ui32): // no predecessors
      %0 = "spn.constant"() {value = 5.00000e-01 : f64} : () -> !spn.probability
      %1 = "spn.constant"() {value = -5.00000e-01 : f64} : () -> !spn.probability
      %2 = "spn.sum"(%0) {opCount = 1 : ui32} : (!spn.probability) -> !spn.probability
      %3 = "spn.sum"(%2) {opCount = 1 : ui32} : (!spn.probability) -> !spn.probability
      %4 = "spn.sum"(%2, %3) {opCount = 2 : ui32} : (!spn.probability, !spn.probability) -> !spn.probability
      %5 = "spn.sum"(%4) {opCount = 1 : ui32} : (!spn.probability) -> !spn.probability
      %6 = "spn.sum"(%5, %1, %1) {opCount = 3 : ui32} : (!spn.probability, !spn.probability, !spn.probability) -> !spn.probability
      "spn.return"(%6) : (!spn.probability) -> ()
  }) {inputType = ui32, kernelName = "spn_kernel", numFeatures = 2 : ui32} : () -> ()

}

//  This small test checks if a sum is reduced if there is only one operand.
//  In the example %6 will be constant (0.0) and can be replaced.
//  This is achieved by multiple folds & reductions, converging at const 0.0.

// CHECK-LABEL: ^bb0
// CHECK-NEXT: %[[#CONST:]] = "spn.constant"()
// CHECK-SAME: {value = 0.000000e+00 : f64}
// CHECK-NEXT: "spn.return"(%[[#CONST]])