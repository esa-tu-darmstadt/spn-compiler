// RUN: %optcall --spn-simplify --canonicalize --spn-type-pinning --spn-to-standard %s | FileCheck %s

module {

  "spn.joint_query"() ( {
     ^bb0(%arg0: f32): // no predecessors
      %0 = "spn.gaussian"(%arg0) {mean = 1.250000e-01 : f64, stddev = 2.500000e-01 : f64} : (f32) -> !spn.probability
      "spn.return"(%0) : (!spn.probability) -> ()
  }) {batchSize = 1 : ui32, errorModel = 1 : i32, inputType = f32, kernelName = "spn_kernel", maxError = 2.000000e-02 : f64, numFeatures = 1 : ui32} : () -> ()

}

// Gaussian distributions get lowered to an actual computation of the Gaussian probability density function (PDF)
// in the Standard dialect.

// CHECK: func @spn_kernel
// CHECK-SAME: %arg[[#ARG1:]]: memref<1xf32>
// CHECK-SAME: %arg[[#ARG2:]]: memref<1xf64>
// CHECK-NEXT: %c[[#CONST0:]]
// CHECK-SAME: constant 0 : index
// CHECK-NEXT: %[[#LOAD:]] = load %arg[[#ARG1]][%c[[#CONST0]]] : memref<1xf32>
// CHECK-DAG: %[[#INPUT:]] = fpext %[[#LOAD]] : f32 to f64
// CHECK-DAG: [[COEFF:%[a-zA-Z_][a-zA-Z0-9_]*]] = constant 1.5957691216057308 : f64
// CHECK-DAG: [[DENOM:%[a-zA-Z_][a-zA-Z0-9_]*]] = constant -8.000000e+00 : f64
// CHECK-DAG: [[MEAN:%[a-zA-Z_][a-zA-Z0-9_]*]] = constant 1.250000e-01 : f64
// CHECK-NOT
// CHECK-DAG: %[[#NOM:]] = subf %[[#INPUT]], [[MEAN]]
// CHECK-NEXT: %[[#NOM_SQUARED:]] = mulf %[[#NOM]], %[[#NOM]]
// CHECK-NEXT: %[[#FRACTION:]] = mulf %[[#NOM_SQUARED]], [[DENOM]]
// CHECK-NEXT: %[[#EXP:]] = exp %[[#FRACTION]]
// CHECK-NEXT: %[[#GAUSSIAN:]] = mulf [[COEFF]], %[[#EXP]]
// CHECK-DAG: %[[#LOG:]] = log %[[#GAUSSIAN]]
// CHECK-NOT: store
// CHECK-DAG: store %[[#LOG]], %arg[[#ARG2]]
// CHECK-NEXT: return