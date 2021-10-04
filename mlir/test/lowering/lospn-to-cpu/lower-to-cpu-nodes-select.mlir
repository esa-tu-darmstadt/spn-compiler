// RUN: %optcall --convert-lospn-nodes-to-cpu %s | FileCheck %s

module  {
  func @task_0(%arg0: memref<?x6xf64>, %arg1: memref<?xf64>) {
    %cst1 = constant 1.000000e-01 : f64
    %ind1 = constant 1 : index
    %0 = "lo_spn.select"(%cst1) {input_true_threshold = 1.000000e+00 : f64, supportMarginal = false, val_false = 5.500000e-01 : f64, val_true = 3.500000e-01 : f64} : (f64) -> f64
    "lo_spn.batch_write"(%0, %arg1, %ind1) : (f64, memref<?xf64>, index) -> ()
    return
  }
}

// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py


// CHECK-LABEL:   func @task_0(
// CHECK-SAME:                 %[[VAL_0:.*]]: memref<?x6xf64>,
// CHECK-SAME:                 %[[VAL_1:.*]]: memref<?xf64>) {
// CHECK:           %[[VAL_2:.*]] = constant 1.000000e-01 : f64
// CHECK:           %[[VAL_3:.*]] = constant 1 : index
// CHECK:           %[[VAL_4:.*]] = constant 0.000000e+00 : f64
// CHECK:           %[[VAL_5:.*]] = constant 2.000000e+00 : f64
// CHECK:           %[[VAL_6:.*]] = cmpf uge, %[[VAL_2]], %[[VAL_4]] : f64
// CHECK:           %[[VAL_7:.*]] = cmpf ult, %[[VAL_2]], %[[VAL_5]] : f64
// CHECK:           %[[VAL_8:.*]] = constant 1.000000e+00 : f64
// CHECK:           %[[VAL_9:.*]] = cmpf ult, %[[VAL_2]], %[[VAL_8]] : f64
// CHECK:           %[[VAL_10:.*]] = constant 3.500000e-01 : f64
// CHECK:           %[[VAL_11:.*]] = constant 5.500000e-01 : f64
// CHECK:           %[[VAL_12:.*]] = select %[[VAL_9]], %[[VAL_10]], %[[VAL_11]] : f64
// CHECK:           %[[VAL_13:.*]] = and %[[VAL_6]], %[[VAL_7]] : i1
// CHECK:           %[[VAL_14:.*]] = constant 0.000000e+00 : f64
// CHECK:           %[[VAL_15:.*]] = select %[[VAL_13]], %[[VAL_12]], %[[VAL_14]] : f64
// CHECK:           memref.store %[[VAL_15]], %[[VAL_1]]{{\[}}%[[VAL_3]]] : memref<?xf64>
// CHECK:           return
// CHECK:         }
