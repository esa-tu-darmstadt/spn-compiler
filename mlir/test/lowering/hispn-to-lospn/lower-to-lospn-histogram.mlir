// RUN: %optcall --convert-hispn-node-to-lospn %s | FileCheck %s

module  {
  "hi_spn.joint_query"() ( {
    "hi_spn.graph"() ( {
    ^bb0(%arg0: i32):  // no predecessors
      %0 = "hi_spn.histogram"(%arg0) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}]} : (i32) -> !hi_spn.probability
      "hi_spn.root"(%0) : (!hi_spn.probability) -> ()
    }) {numFeatures = 1 : ui32} : () -> ()
  }) {batchSize = 12 : ui32, errorModel = 1 : i32, inputType = i32, kernelName = "spn_kernel", maxError = 2.000000e-02 : f64, numFeatures = 1 : ui32, supportMarginal = false} : () -> ()
}

// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py


// CHECK-LABEL:   "hi_spn.joint_query"() ( {
// CHECK:           "hi_spn.graph"() ( {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[VAL_1:.*]] = "lo_spn.histogram"(%[[VAL_0]]) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}], supportMarginal = false} : (i32) -> f64
// CHECK:             %[[VAL_2:.*]] = "lo_spn.log"(%[[VAL_1]]) : (f64) -> f64
// CHECK:             "lo_spn.yield"(%[[VAL_2]]) : (f64) -> ()
// CHECK:           }) {numFeatures = 1 : ui32} : () -> ()
// CHECK:         }) {batchSize = 12 : ui32, errorModel = 1 : i32, inputType = i32, kernelName = "spn_kernel", maxError = 2.000000e-02 : f64, numFeatures = 1 : ui32, supportMarginal = false} : () -> ()
