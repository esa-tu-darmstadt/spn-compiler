// RUN: %optcall --tensor-bufferize --finalizing-bufferize %s | FileCheck %s

module  {
  memref.global "private" constant @histogram_1 : memref<2xf64> = dense<[4.500000e-01, 5.500000e-01]>
  memref.global "private" constant @histogram_0 : memref<2xf64> = dense<[2.500000e-01, 7.500000e-01]>
  func @task_0(%arg0: memref<?x2xi32>, %arg1: memref<?xf64>) {
    %c0 = constant 0 : index
    %c0_0 = constant 0 : index
    %0 = memref.dim %arg0, %c0_0 : memref<?x2xi32>
    %c1 = constant 1 : index
    scf.for %arg2 = %c0 to %0 step %c1 {
      %c0_1 = constant 0 : index
      %1 = memref.load %arg0[%arg2, %c0_1] : memref<?x2xi32>
      %c1_2 = constant 1 : index
      %2 = memref.load %arg0[%arg2, %c1_2] : memref<?x2xi32>
      %3 = memref.get_global @histogram_0 : memref<2xf64>
      %4 = index_cast %1 : i32 to index
      %5 = memref.load %3[%4] : memref<2xf64>
      %6 = memref.get_global @histogram_1 : memref<2xf64>
      %7 = index_cast %2 : i32 to index
      %8 = memref.load %6[%7] : memref<2xf64>
      %9 = mulf %5, %8 : f64
      %10 = math.log %9 : f64
      memref.store %10, %arg1[%arg2] : memref<?xf64>
    }
    return
  }
  func @spn_kernel(%arg0: memref<?x2xi32>, %arg1: memref<?xf64>) {
    %c0 = constant 0 : index
    %0 = memref.dim %arg0, %c0 : memref<?x2xi32>
    %1 = memref.alloc(%0) : memref<?xf64>
    call @task_0(%arg0, %1) : (memref<?x2xi32>, memref<?xf64>) -> ()
    %2 = memref.tensor_load %1 : memref<?xf64>
    %3 = memref.buffer_cast %2 : memref<?xf64>
    "lo_spn.copy"(%3, %arg1) : (memref<?xf64>, memref<?xf64>) -> ()
    return
  }
}

// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

// CHECK-LABEL:   memref.global "private" constant @histogram_1 : memref<2xf64> = dense<[4.500000e-01, 5.500000e-01]>
// CHECK:         memref.global "private" constant @histogram_0 : memref<2xf64> = dense<[2.500000e-01, 7.500000e-01]>

// CHECK-LABEL:   func @task_0(
// CHECK-SAME:                 %[[VAL_0:.*]]: memref<?x2xi32>,
// CHECK-SAME:                 %[[VAL_1:.*]]: memref<?xf64>) {
// CHECK:           %[[VAL_2:.*]] = constant 0 : index
// CHECK:           %[[VAL_3:.*]] = constant 0 : index
// CHECK:           %[[VAL_4:.*]] = memref.dim %[[VAL_0]], %[[VAL_3]] : memref<?x2xi32>
// CHECK:           %[[VAL_5:.*]] = constant 1 : index
// CHECK:           scf.for %[[VAL_6:.*]] = %[[VAL_2]] to %[[VAL_4]] step %[[VAL_5]] {
// CHECK:             %[[VAL_7:.*]] = constant 0 : index
// CHECK:             %[[VAL_8:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_6]], %[[VAL_7]]] : memref<?x2xi32>
// CHECK:             %[[VAL_9:.*]] = constant 1 : index
// CHECK:             %[[VAL_10:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_6]], %[[VAL_9]]] : memref<?x2xi32>
// CHECK:             %[[VAL_11:.*]] = memref.get_global @histogram_0 : memref<2xf64>
// CHECK:             %[[VAL_12:.*]] = index_cast %[[VAL_8]] : i32 to index
// CHECK:             %[[VAL_13:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_12]]] : memref<2xf64>
// CHECK:             %[[VAL_14:.*]] = memref.get_global @histogram_1 : memref<2xf64>
// CHECK:             %[[VAL_15:.*]] = index_cast %[[VAL_10]] : i32 to index
// CHECK:             %[[VAL_16:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_15]]] : memref<2xf64>
// CHECK:             %[[VAL_17:.*]] = mulf %[[VAL_13]], %[[VAL_16]] : f64
// CHECK:             %[[VAL_18:.*]] = math.log %[[VAL_17]] : f64
// CHECK:             memref.store %[[VAL_18]], %[[VAL_1]]{{\[}}%[[VAL_6]]] : memref<?xf64>
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func @spn_kernel(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<?x2xi32>,
// CHECK-SAME:                     %[[VAL_1:.*]]: memref<?xf64>) {
// CHECK:           %[[VAL_2:.*]] = constant 0 : index
// CHECK:           %[[VAL_3:.*]] = memref.dim %[[VAL_0]], %[[VAL_2]] : memref<?x2xi32>
// CHECK:           %[[VAL_4:.*]] = memref.alloc(%[[VAL_3]]) : memref<?xf64>
// CHECK:           call @task_0(%[[VAL_0]], %[[VAL_4]]) : (memref<?x2xi32>, memref<?xf64>) -> ()
// CHECK:           "lo_spn.copy"(%[[VAL_4]], %[[VAL_1]]) : (memref<?xf64>, memref<?xf64>) -> ()
// CHECK:           return
// CHECK:         }
