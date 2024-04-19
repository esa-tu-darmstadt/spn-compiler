// RUN: %optcall --convert-hispn-node-to-lospn %s | FileCheck %s

module  {
  "hi_spn.joint_query"() ( {
    "hi_spn.graph"() ( {
    ^bb0(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32):  // no predecessors
       %0 = "hi_spn.categorical"(%arg0) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01]} : (i32) -> !hi_spn.probability
       %1 = "hi_spn.categorical"(%arg1) {probabilities = [2.500000e-01, 6.250000e-01, 1.250000e-01]} : (i32) -> !hi_spn.probability
       %2 = "hi_spn.categorical"(%arg2) {probabilities = [5.000000e-01, 2.000000e-01, 3.000000e-01]} : (i32) -> !hi_spn.probability
       %3 = "hi_spn.categorical"(%arg3) {probabilities = [6.000000e-01, 1.500000e-01, 2.500000e-01]} : (i32) -> !hi_spn.probability
       %4 = "hi_spn.categorical"(%arg4) {probabilities = [8.000000e-01, 1.400000e-01, 6.000000e-02]} : (i32) -> !hi_spn.probability
       %12 = "hi_spn.sum"(%0, %1, %2, %3, %4) {weights = [0.1,0.1,0.1,0.4,0.3]} : (!hi_spn.probability, !hi_spn.probability, !hi_spn.probability, !hi_spn.probability, !hi_spn.probability) -> !hi_spn.probability
      "hi_spn.root"(%12) : (!hi_spn.probability) -> ()
    }) {numFeatures = 5 : ui32} : () -> ()
  }) {batchSize = 36 : ui32, errorModel = 1 : i32, inputType = i32, kernelName = "spn_kernel", maxError = 2.000000e-02 : f64, numFeatures = 5 : ui32, supportMarginal = false} : () -> ()
}

// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

// CHECK-LABEL:   "hi_spn.joint_query"() ({
// CHECK:           "hi_spn.graph"() ({
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32):
// CHECK:             %[[VAL_5:.*]] = "lo_spn.categorical"(%[[VAL_0]]) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false} : (i32) -> f64
// CHECK:             %[[VAL_6:.*]] = "lo_spn.categorical"(%[[VAL_1]]) {probabilities = [2.500000e-01, 6.250000e-01, 1.250000e-01], supportMarginal = false} : (i32) -> f64
// CHECK:             %[[VAL_7:.*]] = "lo_spn.categorical"(%[[VAL_2]]) {probabilities = [5.000000e-01, 2.000000e-01, 3.000000e-01], supportMarginal = false} : (i32) -> f64
// CHECK:             %[[VAL_8:.*]] = "lo_spn.categorical"(%[[VAL_3]]) {probabilities = [6.000000e-01, 1.500000e-01, 2.500000e-01], supportMarginal = false} : (i32) -> f64
// CHECK:             %[[VAL_9:.*]] = "lo_spn.categorical"(%[[VAL_4]]) {probabilities = [8.000000e-01, 1.400000e-01, 6.000000e-02], supportMarginal = false} : (i32) -> f64
// CHECK:             %[[VAL_10:.*]] = "lo_spn.constant"() {type = f64, value = 1.000000e-01 : f64} : () -> f64
// CHECK:             %[[VAL_11:.*]] = "lo_spn.mul"(%[[VAL_5]], %[[VAL_10]]) : (f64, f64) -> f64
// CHECK:             %[[VAL_12:.*]] = "lo_spn.constant"() {type = f64, value = 1.000000e-01 : f64} : () -> f64
// CHECK:             %[[VAL_13:.*]] = "lo_spn.mul"(%[[VAL_6]], %[[VAL_12]]) : (f64, f64) -> f64
// CHECK:             %[[VAL_14:.*]] = "lo_spn.add"(%[[VAL_11]], %[[VAL_13]]) : (f64, f64) -> f64
// CHECK:             %[[VAL_15:.*]] = "lo_spn.constant"() {type = f64, value = 1.000000e-01 : f64} : () -> f64
// CHECK:             %[[VAL_16:.*]] = "lo_spn.mul"(%[[VAL_7]], %[[VAL_15]]) : (f64, f64) -> f64
// CHECK:             %[[VAL_17:.*]] = "lo_spn.add"(%[[VAL_14]], %[[VAL_16]]) : (f64, f64) -> f64
// CHECK:             %[[VAL_18:.*]] = "lo_spn.constant"() {type = f64, value = 4.000000e-01 : f64} : () -> f64
// CHECK:             %[[VAL_19:.*]] = "lo_spn.mul"(%[[VAL_8]], %[[VAL_18]]) : (f64, f64) -> f64
// CHECK:             %[[VAL_20:.*]] = "lo_spn.constant"() {type = f64, value = 3.000000e-01 : f64} : () -> f64
// CHECK:             %[[VAL_21:.*]] = "lo_spn.mul"(%[[VAL_9]], %[[VAL_20]]) : (f64, f64) -> f64
// CHECK:             %[[VAL_22:.*]] = "lo_spn.add"(%[[VAL_19]], %[[VAL_21]]) : (f64, f64) -> f64
// CHECK:             %[[VAL_23:.*]] = "lo_spn.add"(%[[VAL_17]], %[[VAL_22]]) : (f64, f64) -> f64
// CHECK:             %[[VAL_24:.*]] = "lo_spn.log"(%[[VAL_23]]) : (f64) -> f64
// CHECK:             "lo_spn.yield"(%[[VAL_24]]) : (f64) -> ()
// CHECK:           }) {numFeatures = 5 : ui32} : () -> ()
// CHECK:         }) {batchSize = 36 : ui32, errorModel = 1 : i32, inputType = i32, kernelName = "spn_kernel", maxError = 2.000000e-02 : f64, numFeatures = 5 : ui32, supportMarginal = false} : () -> ()
