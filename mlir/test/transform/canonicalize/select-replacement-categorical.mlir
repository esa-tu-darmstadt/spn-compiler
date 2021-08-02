// RUN: %optcall --canonicalize %s | FileCheck %s

module  {
  "lo_spn.kernel"() ( {
  ^bb0(%arg0: memref<?x1xf64>, %arg1: memref<?xf64>):  // no predecessors
    %c0 = constant 0 : index
    %0 = memref.dim %arg0, %c0 : memref<?x1xf64>
    %1 = memref.alloc(%0) : memref<?xf64>
    "lo_spn.task"(%arg0, %1) ( {
    ^bb0(%arg2: index, %arg3: memref<?x1xf64>, %arg4: memref<?xf64>):  // no predecessors
      %4 = "lo_spn.batch_read"(%arg3, %arg2) {sampleIndex = 0 : ui32} : (memref<?x1xf64>, index) -> f64
      %5 = "lo_spn.body"(%4) ( {
      ^bb0(%arg5: f64):  // no predecessors
        %6 = "lo_spn.categorical"(%arg5) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false} : (f64) -> f64
        %7 = "lo_spn.categorical"(%arg5) {probabilities = [4.500000e-01, 5.500000e-01], supportMarginal = false} : (f64) -> f64
        %8 = "lo_spn.log"(%7) : (f64) -> f64
        "lo_spn.yield"(%8) : (f64) -> ()
      }) : (f64) -> f64
      "lo_spn.batch_write"(%5, %arg4, %arg2) : (f64, memref<?xf64>, index) -> ()
      "lo_spn.return"() : () -> ()
    }) {batchSize = 12 : ui32} : (memref<?x1xf64>, memref<?xf64>) -> ()
    %2 = memref.tensor_load %1 : memref<?xf64>
    %3 = memref.buffer_cast %2 : memref<?xf64>
    "lo_spn.copy"(%3, %arg1) : (memref<?xf64>, memref<?xf64>) -> ()
    "lo_spn.return"() : () -> ()
  }) {sym_name = "spn_kernel", type = (memref<?x1xf64>, memref<?xf64>) -> ()} : () -> ()
}

// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py


// CHECK-LABEL:   "lo_spn.kernel"() ( {
// CHECK:         ^bb0(%[[VAL_0:.*]]: memref<?x1xf64>, %[[VAL_1:.*]]: memref<?xf64>):
// CHECK:           %[[VAL_2:.*]] = constant 0 : index
// CHECK:           %[[VAL_3:.*]] = memref.dim %[[VAL_0]], %[[VAL_2]] : memref<?x1xf64>
// CHECK:           %[[VAL_4:.*]] = memref.alloc(%[[VAL_3]]) : memref<?xf64>
// CHECK:           "lo_spn.task"(%[[VAL_0]], %[[VAL_4]]) ( {
// CHECK:           ^bb0(%[[VAL_5:.*]]: index, %[[VAL_6:.*]]: memref<?x1xf64>, %[[VAL_7:.*]]: memref<?xf64>):
// CHECK:             %[[VAL_8:.*]] = "lo_spn.batch_read"(%[[VAL_6]], %[[VAL_5]]) {sampleIndex = 0 : ui32} : (memref<?x1xf64>, index) -> f64
// CHECK:             %[[VAL_9:.*]] = "lo_spn.body"(%[[VAL_8]]) ( {
// CHECK:             ^bb0(%[[VAL_10:.*]]: f64):
// CHECK:               %[[VAL_11:.*]] = "lo_spn.select"(%[[VAL_10]]) {input_true_threshold = 1.000000e+00 : f64, supportMarginal = false, val_false = 5.500000e-01 : f64, val_true = 4.500000e-01 : f64} : (f64) -> f64
// CHECK:               %[[VAL_12:.*]] = "lo_spn.log"(%[[VAL_11]]) : (f64) -> f64
// CHECK:               "lo_spn.yield"(%[[VAL_12]]) : (f64) -> ()
// CHECK:             }) : (f64) -> f64
// CHECK:             "lo_spn.batch_write"(%[[VAL_13:.*]], %[[VAL_7]], %[[VAL_5]]) : (f64, memref<?xf64>, index) -> ()
// CHECK:             "lo_spn.return"() : () -> ()
// CHECK:           }) {batchSize = 12 : ui32} : (memref<?x1xf64>, memref<?xf64>) -> ()
// CHECK:           "lo_spn.copy"(%[[VAL_4]], %[[VAL_1]]) : (memref<?xf64>, memref<?xf64>) -> ()
// CHECK:           "lo_spn.return"() : () -> ()
// CHECK:         }) {sym_name = "spn_kernel", type = (memref<?x1xf64>, memref<?xf64>) -> ()} : () -> ()