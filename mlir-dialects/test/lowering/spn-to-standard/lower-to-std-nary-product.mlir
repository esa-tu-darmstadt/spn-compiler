// RUN: %optcall --spn-simplify --canonicalize --spn-type-pinning --spn-to-standard %s | FileCheck %s

module {
  "spn.single_joint"() ( {
    ^bb0(%arg0: ui32, %arg1: ui32, %arg2 : ui32, %arg3 : ui32, %arg4 : ui32): // no predecessors
      %0 = "spn.histogram"(%arg0) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}]} : (ui32) -> !spn.probability
      %1 = "spn.histogram"(%arg1) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}]} : (ui32) -> !spn.probability
      %2 = "spn.histogram"(%arg2) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}]} : (ui32) -> !spn.probability
      %3 = "spn.histogram"(%arg3) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}]} : (ui32) -> !spn.probability
      %4 = "spn.histogram"(%arg4) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}]} : (ui32) -> !spn.probability
      %5 = "spn.product"(%0, %1, %2, %3, %4) : (!spn.probability, !spn.probability, !spn.probability, !spn.probability, !spn.probability) -> !spn.probability
      "spn.return"(%5) : (!spn.probability) -> ()
  }) {inputType = ui32, numFeatures = 5 : ui32} : () -> ()
}

//  N-ary products will be converted & lowered into a tree of binary "mulf" insn.
//  This test uses histograms as "non-const" f64 values. We make sure that each
//  value / intermediate product is used and the last product is returned.

// CHECK: func @single_joint
// CHECK-SAME: %arg[[#ARG1:]]: memref<5xui32>
// CHECK-SAME: %arg[[#ARG2:]]: memref<1xf64>

// CHECK-DAG: %[[#INPUT1:]] = load %arg[[#ARG1]]
// CHECK-DAG: %[[#INPUT2:]] = load %arg[[#ARG1]]
// CHECK-DAG: %[[#INPUT3:]] = load %arg[[#ARG1]]
// CHECK-DAG: %[[#INPUT4:]] = load %arg[[#ARG1]]
// CHECK-DAG: %[[#INPUT5:]] = load %arg[[#ARG1]]

// CHECK-DAG: [[HIST1:%[0-9]+]] = "spn.histogram"(%[[#INPUT1]])
// CHECK-DAG: [[HIST2:%[0-9]+]] = "spn.histogram"(%[[#INPUT2]])
// CHECK-DAG: [[HIST3:%[0-9]+]] = "spn.histogram"(%[[#INPUT3]])
// CHECK-DAG: [[HIST4:%[0-9]+]] = "spn.histogram"(%[[#INPUT4]])
// CHECK-DAG: [[HIST5:%[0-9]+]] = "spn.histogram"(%[[#INPUT5]])

// CHECK-DAG: [[PRODUCT_0:%[0-9]+]] = mulf
// CHECK-DAG: [[PRODUCT_1:%[0-9]+]] = mulf
// CHECK-DAG: [[PRODUCT_2:%[0-9]+]] = mulf
// CHECK-DAG: [[PRODUCT_FINAL:%[0-9]+]] = mulf

// CHECK-DAG: [[HIST1]]
// CHECK-DAG: [[HIST2]]
// CHECK-DAG: [[HIST3]]
// CHECK-DAG: [[HIST4]]
// CHECK-DAG: [[HIST5]]
// CHECK-DAG: [[PRODUCT_0]]
// CHECK-DAG: [[PRODUCT_1]]
// CHECK-DAG: [[PRODUCT_2]]

// CHECK-NOT: mulf
// CHECK-DAG: store [[PRODUCT_FINAL]], %arg[[#ARG2]]
// CHECK-NEXT: return

