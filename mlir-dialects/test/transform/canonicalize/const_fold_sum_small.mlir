// RUN: %optcall --canonicalize %s | FileCheck %s

module {

  "spn.single_joint"() ( {
     ^bb0(%arg0: ui32, %arg1: ui32): // no predecessors
      %0 = "spn.constant"() {value = 5.00000e-01 : f64} : () -> !spn.probability
      %1 = "spn.constant"() {value = 2.500000e-01 : f64} : () -> !spn.probability
      %2 = "spn.constant"() {value = 1.250000e-01 : f64} : () -> !spn.probability
      %3 = "spn.constant"() {value = 6.250000e-02 : f64} : () -> !spn.probability
      %4 = "spn.sum"(%0, %1, %2) : (!spn.probability, !spn.probability, !spn.probability) -> !spn.probability
      %5 = "spn.sum"(%3, %3) : (!spn.probability, !spn.probability) -> !spn.probability
      %6 = "spn.sum"(%4, %5) : (!spn.probability, !spn.probability) -> !spn.probability
      "spn.return"(%6) : (!spn.probability) -> ()
  }) {inputType = ui32, numFeatures = 2 : ui32} : () -> ()

}

//  This small test checks if a sum is fold if there are constant operands.
//  In the example %6 will be constant (1.0) and can be replaced.
//  This is achieved by multiple sum-op folds, converging at const 1.0.

// CHECK-LABEL: ^bb0
// CHECK-NEXT: %[[#CONST:]] = "spn.constant"()
// CHECK-SAME: {value = 1.000000e+00 : f64}
// CHECK-NEXT: "spn.return"(%[[#CONST]])
