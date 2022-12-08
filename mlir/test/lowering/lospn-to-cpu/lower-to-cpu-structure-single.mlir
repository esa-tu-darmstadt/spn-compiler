// RUN: %optcall --convert-lospn-structure-to-cpu %s | FileCheck %s

module  {
  "lo_spn.kernel"() ( {
  ^bb0(%arg0: memref<?x2xi32>, %arg1: memref<1x?xf64>):  // no predecessors
    %c0 = arith.constant 0 : index
    %0 = memref.dim %arg0, %c0 : memref<?x2xi32>
    %1 = memref.alloc(%0) : memref<1x?xf64>
    "lo_spn.task"(%arg0, %1) ( {
    ^bb0(%arg2: index, %arg3: memref<?x2xi32>, %arg4: memref<1x?xf64>):  // no predecessors
      %4 = "lo_spn.batch_read"(%arg3, %arg2) {staticIndex = 0 : ui32} : (memref<?x2xi32>, index) -> i32
      %5 = "lo_spn.batch_read"(%arg3, %arg2) {staticIndex = 1 : ui32} : (memref<?x2xi32>, index) -> i32
      %6 = "lo_spn.body"(%4, %5) ( {
      ^bb0(%arg5: i32, %arg6: i32):  // no predecessors
        %7 = "lo_spn.histogram"(%arg5) {bucketCount = 2 : ui32, buckets = [#lo_spn.bucket<0, 1, 2.500000e-01>, #lo_spn.bucket<1, 2, 7.500000e-01>], supportMarginal = false} : (i32) -> f64
        %8 = "lo_spn.histogram"(%arg6) {bucketCount = 2 : ui32, buckets = [#lo_spn.bucket<0, 1, 4.500000e-01>, #lo_spn.bucket<1, 2, 5.500000e-01>], supportMarginal = false} : (i32) -> f64
        %9 = "lo_spn.mul"(%7, %8) : (f64, f64) -> f64
        %18 = "lo_spn.log"(%9) : (f64) -> f64
        "lo_spn.yield"(%18) : (f64) -> ()
      }) : (i32, i32) -> f64
      "lo_spn.batch_write"(%arg4, %arg2, %6) {transposed = true} : (memref<1x?xf64>, index, f64) -> ()
      "lo_spn.return"() : () -> ()
    }) {batchSize = 1 : ui32} : (memref<?x2xi32>, memref<1x?xf64>) -> ()
    //%2 = bufferization.to_tensor %1 : memref<1x?xf64>
    //%3 = bufferization.to_memref %2 : memref<1x?xf64>
    "lo_spn.copy"(%1, %arg1) : (memref<1x?xf64>, memref<1x?xf64>) -> ()
    "lo_spn.return"() : () -> ()
  }) {sym_name = "spn_kernel", function_type = (memref<?x2xi32>, memref<1x?xf64>) -> ()} : () -> ()
}

// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

// The script is designed to make adding checks to
// a test case fast, it is *not* designed to be authoritative
// about what constitutes a good test! The CHECK should be
// minimized and named to reflect the test intent.



// CHECK-LABEL:   func.func @task_0(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<?x2xi32>,
// CHECK-SAME:                      %[[VAL_1:.*]]: memref<1x?xf64>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = "lo_spn.batch_read"(%[[VAL_0]], %[[VAL_2]]) {staticIndex = 0 : ui32} : (memref<?x2xi32>, index) -> i32
// CHECK:           %[[VAL_4:.*]] = "lo_spn.batch_read"(%[[VAL_0]], %[[VAL_2]]) {staticIndex = 1 : ui32} : (memref<?x2xi32>, index) -> i32
// CHECK:           %[[VAL_5:.*]] = "lo_spn.histogram"(%[[VAL_3]]) {bucketCount = 2 : ui32, buckets = [#lo_spn.bucket<0, 1, 2.500000e-01>, #lo_spn.bucket<1, 2, 7.500000e-01>], supportMarginal = false} : (i32) -> f64
// CHECK:           %[[VAL_6:.*]] = "lo_spn.histogram"(%[[VAL_4]]) {bucketCount = 2 : ui32, buckets = [#lo_spn.bucket<0, 1, 4.500000e-01>, #lo_spn.bucket<1, 2, 5.500000e-01>], supportMarginal = false} : (i32) -> f64
// CHECK:           %[[VAL_7:.*]] = "lo_spn.mul"(%[[VAL_5]], %[[VAL_6]]) : (f64, f64) -> f64
// CHECK:           %[[VAL_8:.*]] = "lo_spn.log"(%[[VAL_7]]) : (f64) -> f64
// CHECK:           "lo_spn.batch_write"(%[[VAL_1]], %[[VAL_2]], %[[VAL_8]]) {transposed = true} : (memref<1x?xf64>, index, f64) -> ()
// CHECK:           "lo_spn.return"() : () -> ()
// CHECK:         }

// CHECK-LABEL:   func.func @spn_kernel(
// CHECK-SAME:                          %[[VAL_0:.*]]: memref<?x2xi32>,
// CHECK-SAME:                          %[[VAL_1:.*]]: memref<1x?xf64>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = memref.dim %[[VAL_0]], %[[VAL_2]] : memref<?x2xi32>
// CHECK:           %[[VAL_4:.*]] = memref.alloc(%[[VAL_3]]) : memref<1x?xf64>
// CHECK:           call @task_0(%[[VAL_0]], %[[VAL_4]]) : (memref<?x2xi32>, memref<1x?xf64>) -> ()
// CHECK:           "lo_spn.copy"(%[[VAL_4]], %[[VAL_1]]) : (memref<1x?xf64>, memref<1x?xf64>) -> ()
// CHECK:           "lo_spn.return"() : () -> ()
// CHECK:         }

