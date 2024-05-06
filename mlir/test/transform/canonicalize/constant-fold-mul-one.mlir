// RUN: %optcall --canonicalize %s | FileCheck %s
// Test (re)generated by regenerate_tests.py.
module {
  "lo_spn.kernel"() <{function_type = (memref<?x2xf64>, memref<1x?xf64>) -> (), sym_name = "spn_cpu"}> ({
  ^bb0(%arg0: memref<?x2xf64>, %arg1: memref<1x?xf64>):
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x2xf64>
    %alloc = memref.alloc(%dim) : memref<1x?xf64>
    "lo_spn.task"(%arg0, %alloc) <{batchSize = 1 : ui32}> ({
    ^bb0(%arg2: index, %arg3: memref<?x2xf64>, %arg4: memref<1x?xf64>):
      %0 = "lo_spn.batch_read"(%arg3, %arg2) <{staticIndex = 0 : ui32}> : (memref<?x2xf64>, index) -> f64
      %1 = "lo_spn.batch_read"(%arg3, %arg2) <{staticIndex = 1 : ui32}> : (memref<?x2xf64>, index) -> f64
      %2 = "lo_spn.body"(%0, %1) ({
      ^bb0(%arg5: f64, %arg6: f64):
        %3 = "lo_spn.mul"(%arg5, %arg6) : (f64, f64) -> f64
        %4 = "lo_spn.constant"() <{value = 1.000000e+00 : f64}> : () -> f64
        %5 = "lo_spn.mul"(%3, %4) : (f64, f64) -> f64
        %6 = "lo_spn.constant"() <{value = 2.000000e+00 : f64}> : () -> f64
        %7 = "lo_spn.constant"() <{value = 3.000000e+00 : f64}> : () -> f64
        %8 = "lo_spn.mul"(%6, %7) : (f64, f64) -> f64
        %9 = "lo_spn.mul"(%5, %8) : (f64, f64) -> f64
        "lo_spn.yield"(%9) : (f64) -> ()
      }) : (f64, f64) -> f64
      "lo_spn.batch_write"(%arg4, %arg2, %2) <{transposed = true}> : (memref<1x?xf64>, index, f64) -> ()
      "lo_spn.return"() : () -> ()
    }) : (memref<?x2xf64>, memref<1x?xf64>) -> ()
    "lo_spn.copy"(%alloc, %arg1) : (memref<1x?xf64>, memref<1x?xf64>) -> ()
    "lo_spn.return"() : () -> ()
  }) {type = (memref<?x2xf64>, memref<1x?xf64>) -> ()} : () -> ()
}

// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

// The script is designed to make adding checks to
// a test case fast, it is *not* designed to be authoritative
// about what constitutes a good test! The CHECK should be
// minimized and named to reflect the test intent.



// CHECK-LABEL:   "lo_spn.kernel"() <{function_type = (memref<?x2xf64>, memref<1x?xf64>) -> (), sym_name = "spn_cpu"}> ({
// CHECK:         ^bb0(%[[VAL_0:.*]]: memref<?x2xf64>, %[[VAL_1:.*]]: memref<1x?xf64>):
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = memref.dim %[[VAL_0]], %[[VAL_2]] : memref<?x2xf64>
// CHECK:           %[[VAL_4:.*]] = memref.alloc(%[[VAL_3]]) : memref<1x?xf64>
// CHECK:           "lo_spn.task"(%[[VAL_0]], %[[VAL_4]]) <{batchSize = 1 : ui32}> ({
// CHECK:           ^bb0(%[[VAL_5:.*]]: index, %[[VAL_6:.*]]: memref<?x2xf64>, %[[VAL_7:.*]]: memref<1x?xf64>):
// CHECK:             %[[VAL_8:.*]] = "lo_spn.batch_read"(%[[VAL_6]], %[[VAL_5]]) <{staticIndex = 0 : ui32}> : (memref<?x2xf64>, index) -> f64
// CHECK:             %[[VAL_9:.*]] = "lo_spn.batch_read"(%[[VAL_6]], %[[VAL_5]]) <{staticIndex = 1 : ui32}> : (memref<?x2xf64>, index) -> f64
// CHECK:             %[[VAL_10:.*]] = "lo_spn.body"(%[[VAL_8]], %[[VAL_9]]) ({
// CHECK:             ^bb0(%[[VAL_11:.*]]: f64, %[[VAL_12:.*]]: f64):
// CHECK:               %[[VAL_13:.*]] = "lo_spn.constant"() <{value = 6.000000e+00 : f64}> : () -> f64
// CHECK:               %[[VAL_14:.*]] = "lo_spn.mul"(%[[VAL_11]], %[[VAL_12]]) : (f64, f64) -> f64
// CHECK:               %[[VAL_15:.*]] = "lo_spn.mul"(%[[VAL_14]], %[[VAL_13]]) : (f64, f64) -> f64
// CHECK:               "lo_spn.yield"(%[[VAL_15]]) : (f64) -> ()
// CHECK:             }) : (f64, f64) -> f64
// CHECK:             "lo_spn.batch_write"(%[[VAL_7]], %[[VAL_5]], %[[VAL_10]]) <{transposed = true}> : (memref<1x?xf64>, index, f64) -> ()
// CHECK:             "lo_spn.return"() : () -> ()
// CHECK:           }) : (memref<?x2xf64>, memref<1x?xf64>) -> ()
// CHECK:           "lo_spn.copy"(%[[VAL_4]], %[[VAL_1]]) : (memref<1x?xf64>, memref<1x?xf64>) -> ()
// CHECK:           "lo_spn.return"() : () -> ()
// CHECK:         }) {type = (memref<?x2xf64>, memref<1x?xf64>) -> ()} : () -> ()

