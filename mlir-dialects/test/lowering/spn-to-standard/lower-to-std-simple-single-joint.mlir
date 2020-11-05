// RUN: %optcall --spn-simplify --canonicalize --spn-type-pinning --spn-to-standard %s | FileCheck %s

module {

  "spn.single_joint"() ( {
     ^bb0(%arg0: ui32, %arg1: ui32): // no predecessors
      %0 = "spn.constant"() {value = 5.00000e-01 : f64} : () -> !spn.probability
      "spn.return"(%0) : (!spn.probability) -> ()
  }) {inputType = ui32, numFeatures = 2 : ui32} : () -> ()

}

// Basic lowering of a single joint query into a function reading the arguments and storing the return value.

// CHECK: func @single_joint
// CHECK-SAME: %arg[[#ARG1:]]: memref<2xui32>
// CHECK-SAME: %arg[[#ARG2:]]: memref<1xf64>
// CHECK-NEXT: %c[[#CONST0:]]
// CHECK-SAME: constant 0 : index
// CHECK-NEXT: load %arg[[#ARG1]][%c[[#CONST0]]] : memref<2xui32>
// CHECK-NEXT: %c[[#CONST1:]]
// CHECK-SAME: constant 1 : index
// CHECK-NEXT: load %arg[[#ARG1]][%c[[#CONST1]]] : memref<2xui32>
// CHECK-NEXT: %[[CONSTANT:[a-zA-Z_][a-zA-Z0-9_]*]]
// CHECK-SAME: constant 5.000000e-01 : f64
// CHECK-NEXT: %[[INDEX:[a-zA-Z_][a-zA-Z0-9_]*]]
// CHECK-SAME: constant 0 : index
// CHECK-NEXT: store %[[CONSTANT]], %arg[[#ARG2]][%[[INDEX]]] : memref<1xf64>


