// RUN: %optcall --lospn-copy-removal %s | FileCheck %s

module  {
  "lo_spn.kernel"() ( {
  ^bb0(%arg0: memref<?x2xi32>, %arg1: memref<1x?xf64>):  // no predecessors
    %c0 = arith.constant 0 : index
    %0 = memref.dim %arg0, %c0 : memref<?x2xi32>
    %1 = memref.alloc(%0) : memref<1x?xf64>
    "lo_spn.task"(%arg0, %1) ( {
    ^bb0(%arg2: index, %arg3: memref<?x2xi32>, %arg4: memref<1x?xf64>):  // no predecessors
      %4 = "lo_spn.batch_read"(%arg3, %arg2) {staticIndex = 0 : ui32, transposed = false} : (memref<?x2xi32>, index) -> i32
      %5 = "lo_spn.batch_read"(%arg3, %arg2) {staticIndex = 1 : ui32, transposed = false} : (memref<?x2xi32>, index) -> i32
      %6 = "lo_spn.body"(%4, %5) ( {
      ^bb0(%arg5: i32, %arg6: i32):  // no predecessors
        %7 = "lo_spn.histogram"(%arg5) {bucketCount = 2 : ui32, buckets = [#lo_spn.bucket<0, 1, 2.500000e-01>, #lo_spn.bucket<1, 2, 7.500000e-01>], supportMarginal = false} : (i32) -> f64
        %8 = "lo_spn.histogram"(%arg6) {bucketCount = 2 : ui32, buckets = [#lo_spn.bucket<0, 1, 4.500000e-01>, #lo_spn.bucket<1, 2, 5.500000e-01>], supportMarginal = false} : (i32) -> f64
        %9 = "lo_spn.mul"(%7, %8) : (f64, f64) -> f64
        %10 = "lo_spn.histogram"(%arg5) {bucketCount = 2 : ui32, buckets = [#lo_spn.bucket<0, 1, 3.300000e-01>, #lo_spn.bucket<1, 2, 6.700000e-01>], supportMarginal = false} : (i32) -> f64
        %11 = "lo_spn.histogram"(%arg6) {bucketCount = 2 : ui32, buckets = [#lo_spn.bucket<0, 1, 8.750000e-01>, #lo_spn.bucket<1, 2, 1.250000e-01>], supportMarginal = false} : (i32) -> f64
        %12 = "lo_spn.mul"(%10, %11) : (f64, f64) -> f64
        %13 = "lo_spn.constant"() {type = f64, value = 3.000000e-01 : f64} : () -> f64
        %14 = "lo_spn.mul"(%9, %13) : (f64, f64) -> f64
        %15 = "lo_spn.constant"() {type = f64, value = 0.69999999999999996 : f64} : () -> f64
        %16 = "lo_spn.mul"(%12, %15) : (f64, f64) -> f64
        %17 = "lo_spn.add"(%14, %16) : (f64, f64) -> f64
        %18 = "lo_spn.log"(%17) : (f64) -> f64
        "lo_spn.yield"(%18) : (f64) -> ()
      }) : (i32, i32) -> f64
      "lo_spn.batch_write"(%arg4, %arg2, %6) {transposed = true} : (memref<1x?xf64>, index, f64) -> ()
      "lo_spn.return"() : () -> ()
    }) {batchSize = 10 : ui32} : (memref<?x2xi32>, memref<1x?xf64>) -> ()
    %2 = bufferization.to_tensor %1 : memref<1x?xf64>
    %3 = bufferization.to_memref %2 : memref<1x?xf64>
    "lo_spn.copy"(%3, %arg1) : (memref<1x?xf64>, memref<1x?xf64>) -> ()
    "lo_spn.return"() : () -> ()
  }) {sym_name = "spn_kernel", function_type = (memref<?x2xi32>, memref<1x?xf64>) -> ()} : () -> ()
}



// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

// The script is designed to make adding checks to
// a test case fast, it is *not* designed to be authoritative
// about what constitutes a good test! The CHECK should be
// minimized and named to reflect the test intent.



// CHECK-LABEL:   "lo_spn.kernel"() ({
// CHECK:         ^bb0(%[[VAL_0:.*]]: memref<?x2xi32>, %[[VAL_1:.*]]: memref<1x?xf64>):
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = memref.dim %[[VAL_0]], %[[VAL_2]] : memref<?x2xi32>
// CHECK:           %[[VAL_4:.*]] = memref.alloc(%[[VAL_3]]) : memref<1x?xf64>
// CHECK:           "lo_spn.task"(%[[VAL_0]], %[[VAL_4]]) ({
// CHECK:           ^bb0(%[[VAL_5:.*]]: index, %[[VAL_6:.*]]: memref<?x2xi32>, %[[VAL_7:.*]]: memref<1x?xf64>):
// CHECK:             %[[VAL_8:.*]] = "lo_spn.batch_read"(%[[VAL_6]], %[[VAL_5]]) {staticIndex = 0 : ui32, transposed = false} : (memref<?x2xi32>, index) -> i32
// CHECK:             %[[VAL_9:.*]] = "lo_spn.batch_read"(%[[VAL_6]], %[[VAL_5]]) {staticIndex = 1 : ui32, transposed = false} : (memref<?x2xi32>, index) -> i32
// CHECK:             %[[VAL_10:.*]] = "lo_spn.body"(%[[VAL_8]], %[[VAL_9]]) ({
// CHECK:             ^bb0(%[[VAL_11:.*]]: i32, %[[VAL_12:.*]]: i32):
// CHECK:               %[[VAL_13:.*]] = "lo_spn.constant"() {type = f64, value = 0.69999999999999996 : f64} : () -> f64
// CHECK:               %[[VAL_14:.*]] = "lo_spn.constant"() {type = f64, value = 3.000000e-01 : f64} : () -> f64
// CHECK:               %[[VAL_15:.*]] = "lo_spn.histogram"(%[[VAL_11]]) {bucketCount = 2 : ui32, buckets = [#lo_spn.bucket<0, 1, 2.500000e-01>, #lo_spn.bucket<1, 2, 7.500000e-01>], supportMarginal = false} : (i32) -> f64
// CHECK:               %[[VAL_16:.*]] = "lo_spn.histogram"(%[[VAL_12]]) {bucketCount = 2 : ui32, buckets = [#lo_spn.bucket<0, 1, 4.500000e-01>, #lo_spn.bucket<1, 2, 5.500000e-01>], supportMarginal = false} : (i32) -> f64
// CHECK:               %[[VAL_17:.*]] = "lo_spn.mul"(%[[VAL_15]], %[[VAL_16]]) : (f64, f64) -> f64
// CHECK:               %[[VAL_18:.*]] = "lo_spn.histogram"(%[[VAL_11]]) {bucketCount = 2 : ui32, buckets = [#lo_spn.bucket<0, 1, 3.300000e-01>, #lo_spn.bucket<1, 2, 6.700000e-01>], supportMarginal = false} : (i32) -> f64
// CHECK:               %[[VAL_19:.*]] = "lo_spn.histogram"(%[[VAL_12]]) {bucketCount = 2 : ui32, buckets = [#lo_spn.bucket<0, 1, 8.750000e-01>, #lo_spn.bucket<1, 2, 1.250000e-01>], supportMarginal = false} : (i32) -> f64
// CHECK:               %[[VAL_20:.*]] = "lo_spn.mul"(%[[VAL_18]], %[[VAL_19]]) : (f64, f64) -> f64
// CHECK:               %[[VAL_21:.*]] = "lo_spn.mul"(%[[VAL_17]], %[[VAL_14]]) : (f64, f64) -> f64
// CHECK:               %[[VAL_22:.*]] = "lo_spn.mul"(%[[VAL_20]], %[[VAL_13]]) : (f64, f64) -> f64
// CHECK:               %[[VAL_23:.*]] = "lo_spn.add"(%[[VAL_21]], %[[VAL_22]]) : (f64, f64) -> f64
// CHECK:               %[[VAL_24:.*]] = "lo_spn.log"(%[[VAL_23]]) : (f64) -> f64
// CHECK:               "lo_spn.yield"(%[[VAL_24]]) : (f64) -> ()
// CHECK:             }) : (i32, i32) -> f64
// CHECK:             "lo_spn.batch_write"(%[[VAL_7]], %[[VAL_5]], %[[VAL_25:.*]]) {transposed = true} : (memref<1x?xf64>, index, f64) -> ()
// CHECK:             "lo_spn.return"() : () -> ()
// CHECK:           }) {batchSize = 10 : ui32} : (memref<?x2xi32>, memref<1x?xf64>) -> ()
// CHECK:           "lo_spn.return"() : () -> ()
// CHECK:         }) {function_type = (memref<?x2xi32>, memref<1x?xf64>) -> (), sym_name = "spn_kernel"} : () -> ()
