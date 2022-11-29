// RUN: %optcall --convert-hispn-node-to-lospn --convert-hispn-query-to-lospn %s | FileCheck %s

module  {
  "hi_spn.joint_query"() ( {
    "hi_spn.graph"() ( {
    ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
      %0 = "hi_spn.histogram"(%arg0) {bucketCount = 2 : ui32, buckets = [#hi_spn.bucket<0, 1, 2.500000e-01>, #hi_spn.bucket<1, 2, 7.500000e-01>]} : (i32) -> !hi_spn.probability
      %1 = "hi_spn.histogram"(%arg1) {bucketCount = 2 : ui32, buckets = [#hi_spn.bucket<0, 1, 4.500000e-01>, #hi_spn.bucket<1, 2, 5.500000e-01>]} : (i32) -> !hi_spn.probability
      %2 = "hi_spn.product"(%0, %1) : (!hi_spn.probability, !hi_spn.probability) -> !hi_spn.probability
      %3 = "hi_spn.histogram"(%arg0) {bucketCount = 2 : ui32, buckets = [#hi_spn.bucket<0, 1, 3.300000e-01>, #hi_spn.bucket<1, 2, 6.700000e-01>]} : (i32) -> !hi_spn.probability
      %4 = "hi_spn.histogram"(%arg1) {bucketCount = 2 : ui32, buckets = [#hi_spn.bucket<0, 1, 8.750000e-01>, #hi_spn.bucket<1, 2, 1.250000e-01>]} : (i32) -> !hi_spn.probability
      %5 = "hi_spn.product"(%3, %4) : (!hi_spn.probability, !hi_spn.probability) -> !hi_spn.probability
      %6 = "hi_spn.sum"(%2, %5) {weights = [3.000000e-01, 0.69999999999999996]} : (!hi_spn.probability, !hi_spn.probability) -> !hi_spn.probability
      "hi_spn.root"(%6) : (!hi_spn.probability) -> ()
    }) {numFeatures = 2 : ui32} : () -> ()
  }) {batchSize = 10 : ui32, errorModel = 1 : i32, inputType = i32, kernelName = "spn_kernel", maxError = 2.000000e-02 : f64, numFeatures = 2 : ui32, supportMarginal = false} : () -> ()
}

// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py


// CHECK-LABEL:   "lo_spn.kernel"() ({
// CHECK:         ^bb0(%[[VAL_0:.*]]: tensor<?x2xi32>):
// CHECK:           %[[VAL_1:.*]] = "lo_spn.task"(%[[VAL_0]]) ({
// CHECK:           ^bb0(%[[VAL_2:.*]]: index, %[[VAL_3:.*]]: tensor<?x2xi32>):
// CHECK:             %[[VAL_4:.*]] = "lo_spn.batch_extract"(%[[VAL_3]], %[[VAL_2]]) {staticIndex = 0 : ui32, transposed = false} : (tensor<?x2xi32>, index) -> i32
// CHECK:             %[[VAL_5:.*]] = "lo_spn.batch_extract"(%[[VAL_3]], %[[VAL_2]]) {staticIndex = 1 : ui32, transposed = false} : (tensor<?x2xi32>, index) -> i32
// CHECK:             %[[VAL_6:.*]] = "lo_spn.body"(%[[VAL_4]], %[[VAL_5]]) ({
// CHECK:             ^bb0(%[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32):
// CHECK:               %[[VAL_9:.*]] = "lo_spn.histogram"(%[[VAL_7]]) {bucketCount = 2 : ui32, buckets = [#lo_spn.bucket<0, 1, 2.500000e-01>, #lo_spn.bucket<1, 2, 7.500000e-01>], supportMarginal = false} : (i32) -> f64
// CHECK:               %[[VAL_10:.*]] = "lo_spn.histogram"(%[[VAL_8]]) {bucketCount = 2 : ui32, buckets = [#lo_spn.bucket<0, 1, 4.500000e-01>, #lo_spn.bucket<1, 2, 5.500000e-01>], supportMarginal = false} : (i32) -> f64
// CHECK:               %[[VAL_11:.*]] = "lo_spn.mul"(%[[VAL_9]], %[[VAL_10]]) : (f64, f64) -> f64
// CHECK:               %[[VAL_12:.*]] = "lo_spn.histogram"(%[[VAL_7]]) {bucketCount = 2 : ui32, buckets = [#lo_spn.bucket<0, 1, 3.300000e-01>, #lo_spn.bucket<1, 2, 6.700000e-01>], supportMarginal = false} : (i32) -> f64
// CHECK:               %[[VAL_13:.*]] = "lo_spn.histogram"(%[[VAL_8]]) {bucketCount = 2 : ui32, buckets = [#lo_spn.bucket<0, 1, 8.750000e-01>, #lo_spn.bucket<1, 2, 1.250000e-01>], supportMarginal = false} : (i32) -> f64
// CHECK:               %[[VAL_14:.*]] = "lo_spn.mul"(%[[VAL_12]], %[[VAL_13]]) : (f64, f64) -> f64
// CHECK:               %[[VAL_15:.*]] = "lo_spn.constant"() {type = f64, value = 3.000000e-01 : f64} : () -> f64
// CHECK:               %[[VAL_16:.*]] = "lo_spn.mul"(%[[VAL_11]], %[[VAL_15]]) : (f64, f64) -> f64
// CHECK:               %[[VAL_17:.*]] = "lo_spn.constant"() {type = f64, value = 0.69999999999999996 : f64} : () -> f64
// CHECK:               %[[VAL_18:.*]] = "lo_spn.mul"(%[[VAL_14]], %[[VAL_17]]) : (f64, f64) -> f64
// CHECK:               %[[VAL_19:.*]] = "lo_spn.add"(%[[VAL_16]], %[[VAL_18]]) : (f64, f64) -> f64
// CHECK:               %[[VAL_20:.*]] = "lo_spn.log"(%[[VAL_19]]) : (f64) -> f64
// CHECK:               "lo_spn.yield"(%[[VAL_20]]) : (f64) -> ()
// CHECK:             }) : (i32, i32) -> f64
// CHECK:             %[[VAL_21:.*]] = "lo_spn.batch_collect"(%[[VAL_2]], %[[VAL_22:.*]]) {transposed = true} : (index, f64) -> tensor<1x?xf64>
// CHECK:             "lo_spn.return"(%[[VAL_21]]) : (tensor<1x?xf64>) -> ()
// CHECK:           }) {batchSize = 10 : ui32} : (tensor<?x2xi32>) -> tensor<1x?xf64>
// CHECK:           "lo_spn.return"(%[[VAL_23:.*]]) : (tensor<1x?xf64>) -> ()
// CHECK:         }) {function_type = (tensor<?x2xi32>) -> tensor<1x?xf64>, sym_name = "spn_kernel"} : () -> ()
