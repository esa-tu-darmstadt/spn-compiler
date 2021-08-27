// RUN: %optcall --convert-lospn-nodes-to-cpu %s | FileCheck %s

module  {
  memref.global "private" constant @histogram_vec_1 : memref<2xf64> = dense<[4.500000e-01, 5.500000e-01]>
  memref.global "private" constant @histogram_vec_0 : memref<2xf64> = dense<[2.500000e-01, 7.500000e-01]>
  memref.global "private" constant @categorical_vec_1 : memref<3xf64> = dense<[2.500000e-01, 6.250000e-01, 1.250000e-01]>
  memref.global "private" constant @categorical_vec_0 : memref<3xf64> = dense<[3.500000e-01, 5.500000e-01, 1.000000e-01]>
  func @vec_task_0(%arg0: memref<?x6xf64>, %arg1: memref<1x?xf64>) {
    %c0 = constant 0 : index
    %0 = memref.dim %arg0, %c0 : memref<?x6xf64>
    %c4 = constant 4 : index
    %1 = remi_unsigned %0, %c4 : index
    %2 = subi %0, %1 : index
    %c0_0 = constant 0 : index
    %c4_1 = constant 4 : index
    scf.for %arg2 = %c0_0 to %2 step %c4_1 {
      %3 = index_cast %arg2 : index to i64
      %4 = vector.broadcast %3 : i64 to vector<4xi64>
      %cst = constant dense<[0, 6, 12, 18]> : vector<4xi64>
      %cst_2 = constant dense<6> : vector<4xi64>
      %5 = muli %4, %cst_2 : vector<4xi64>
      %6 = addi %5, %cst : vector<4xi64>
      %cst_3 = constant dense<0.000000e+00> : vector<4xf64>
      %cst_4 = constant dense<true> : vector<4xi1>
      %c0_5 = constant 0 : index
      %7 = memref.dim %arg0, %c0_5 : memref<?x6xf64>
      %c6 = constant 6 : index
      %8 = muli %7, %c6 : index
      %9 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%8], strides: [1] : memref<?x6xf64> to memref<?xf64>
      %c0_6 = constant 0 : index
      %10 = vector.gather %9[%c0_6] [%6], %cst_4, %cst_3 : memref<?xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
      %11 = index_cast %arg2 : index to i64
      %12 = vector.broadcast %11 : i64 to vector<4xi64>
      %cst_7 = constant dense<[1, 7, 13, 19]> : vector<4xi64>
      %cst_8 = constant dense<6> : vector<4xi64>
      %13 = muli %12, %cst_8 : vector<4xi64>
      %14 = addi %13, %cst_7 : vector<4xi64>
      %cst_9 = constant dense<0.000000e+00> : vector<4xf64>
      %cst_10 = constant dense<true> : vector<4xi1>
      %c0_11 = constant 0 : index
      %15 = memref.dim %arg0, %c0_11 : memref<?x6xf64>
      %c6_12 = constant 6 : index
      %16 = muli %15, %c6_12 : index
      %17 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%16], strides: [1] : memref<?x6xf64> to memref<?xf64>
      %c0_13 = constant 0 : index
      %18 = vector.gather %17[%c0_13] [%14], %cst_10, %cst_9 : memref<?xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
      %19 = index_cast %arg2 : index to i64
      %20 = vector.broadcast %19 : i64 to vector<4xi64>
      %cst_14 = constant dense<[2, 8, 14, 20]> : vector<4xi64>
      %cst_15 = constant dense<6> : vector<4xi64>
      %21 = muli %20, %cst_15 : vector<4xi64>
      %22 = addi %21, %cst_14 : vector<4xi64>
      %cst_16 = constant dense<0.000000e+00> : vector<4xf64>
      %cst_17 = constant dense<true> : vector<4xi1>
      %c0_18 = constant 0 : index
      %23 = memref.dim %arg0, %c0_18 : memref<?x6xf64>
      %c6_19 = constant 6 : index
      %24 = muli %23, %c6_19 : index
      %25 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%24], strides: [1] : memref<?x6xf64> to memref<?xf64>
      %c0_20 = constant 0 : index
      %26 = vector.gather %25[%c0_20] [%22], %cst_17, %cst_16 : memref<?xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
      %27 = index_cast %arg2 : index to i64
      %28 = vector.broadcast %27 : i64 to vector<4xi64>
      %cst_21 = constant dense<[3, 9, 15, 21]> : vector<4xi64>
      %cst_22 = constant dense<6> : vector<4xi64>
      %29 = muli %28, %cst_22 : vector<4xi64>
      %30 = addi %29, %cst_21 : vector<4xi64>
      %cst_23 = constant dense<0.000000e+00> : vector<4xf64>
      %cst_24 = constant dense<true> : vector<4xi1>
      %c0_25 = constant 0 : index
      %31 = memref.dim %arg0, %c0_25 : memref<?x6xf64>
      %c6_26 = constant 6 : index
      %32 = muli %31, %c6_26 : index
      %33 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%32], strides: [1] : memref<?x6xf64> to memref<?xf64>
      %c0_27 = constant 0 : index
      %34 = vector.gather %33[%c0_27] [%30], %cst_24, %cst_23 : memref<?xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
      %35 = index_cast %arg2 : index to i64
      %36 = vector.broadcast %35 : i64 to vector<4xi64>
      %cst_28 = constant dense<[4, 10, 16, 22]> : vector<4xi64>
      %cst_29 = constant dense<6> : vector<4xi64>
      %37 = muli %36, %cst_29 : vector<4xi64>
      %38 = addi %37, %cst_28 : vector<4xi64>
      %cst_30 = constant dense<0.000000e+00> : vector<4xf64>
      %cst_31 = constant dense<true> : vector<4xi1>
      %c0_32 = constant 0 : index
      %39 = memref.dim %arg0, %c0_32 : memref<?x6xf64>
      %c6_33 = constant 6 : index
      %40 = muli %39, %c6_33 : index
      %41 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%40], strides: [1] : memref<?x6xf64> to memref<?xf64>
      %c0_34 = constant 0 : index
      %42 = vector.gather %41[%c0_34] [%38], %cst_31, %cst_30 : memref<?xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
      %43 = index_cast %arg2 : index to i64
      %44 = vector.broadcast %43 : i64 to vector<4xi64>
      %cst_35 = constant dense<[5, 11, 17, 23]> : vector<4xi64>
      %cst_36 = constant dense<6> : vector<4xi64>
      %45 = muli %44, %cst_36 : vector<4xi64>
      %46 = addi %45, %cst_35 : vector<4xi64>
      %cst_37 = constant dense<0.000000e+00> : vector<4xf64>
      %cst_38 = constant dense<true> : vector<4xi1>
      %c0_39 = constant 0 : index
      %47 = memref.dim %arg0, %c0_39 : memref<?x6xf64>
      %c6_40 = constant 6 : index
      %48 = muli %47, %c6_40 : index
      %49 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%48], strides: [1] : memref<?x6xf64> to memref<?xf64>
      %c0_41 = constant 0 : index
      %50 = vector.gather %49[%c0_41] [%46], %cst_38, %cst_37 : memref<?xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
      %51 = memref.get_global @categorical_vec_0 : memref<3xf64>
      %52 = fptoui %10 : vector<4xf64> to vector<4xi64>
      %cst_42 = constant dense<0.000000e+00> : vector<4xf64>
      %cst_43 = constant dense<true> : vector<4xi1>
      %c0_44 = constant 0 : index
      %53 = vector.gather %51[%c0_44] [%52], %cst_43, %cst_42 : memref<3xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
      %54 = memref.get_global @categorical_vec_1 : memref<3xf64>
      %55 = fptoui %18 : vector<4xf64> to vector<4xi64>
      %cst_45 = constant dense<0.000000e+00> : vector<4xf64>
      %cst_46 = constant dense<true> : vector<4xi1>
      %c0_47 = constant 0 : index
      %56 = vector.gather %54[%c0_47] [%55], %cst_46, %cst_45 : memref<3xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
      %57 = memref.get_global @histogram_vec_0 : memref<2xf64>
      %58 = fptoui %26 : vector<4xf64> to vector<4xi64>
      %cst_48 = constant dense<0.000000e+00> : vector<4xf64>
      %cst_49 = constant dense<true> : vector<4xi1>
      %c0_50 = constant 0 : index
      %59 = vector.gather %57[%c0_50] [%58], %cst_49, %cst_48 : memref<2xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
      %60 = memref.get_global @histogram_vec_1 : memref<2xf64>
      %61 = fptoui %34 : vector<4xf64> to vector<4xi64>
      %cst_51 = constant dense<0.000000e+00> : vector<4xf64>
      %cst_52 = constant dense<true> : vector<4xi1>
      %c0_53 = constant 0 : index
      %62 = vector.gather %60[%c0_53] [%61], %cst_52, %cst_51 : memref<2xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
      %cst_54 = constant dense<0.3989422804014327> : vector<4xf64>
      %cst_55 = constant dense<-5.000000e-01> : vector<4xf64>
      %cst_56 = constant dense<5.000000e-01> : vector<4xf64>
      %63 = subf %42, %cst_56 : vector<4xf64>
      %64 = mulf %63, %63 : vector<4xf64>
      %65 = mulf %64, %cst_55 : vector<4xf64>
      %66 = math.exp %65 : vector<4xf64>
      %67 = mulf %cst_54, %66 : vector<4xf64>
      %cst_57 = constant dense<3.9894228040143269> : vector<4xf64>
      %cst_58 = constant dense<-49.999999999999993> : vector<4xf64>
      %cst_59 = constant dense<2.500000e-01> : vector<4xf64>
      %68 = subf %50, %cst_59 : vector<4xf64>
      %69 = mulf %68, %68 : vector<4xf64>
      %70 = mulf %69, %cst_58 : vector<4xf64>
      %71 = math.exp %70 : vector<4xf64>
      %72 = mulf %cst_57, %71 : vector<4xf64>
      %73 = mulf %53, %56 : vector<4xf64>
      %74 = mulf %73, %59 : vector<4xf64>
      %cst_60 = constant dense<1.000000e-01> : vector<4xf64>
      %75 = mulf %74, %cst_60 : vector<4xf64>
      %76 = mulf %62, %67 : vector<4xf64>
      %77 = mulf %76, %72 : vector<4xf64>
      %cst_61 = constant dense<1.000000e-01> : vector<4xf64>
      %78 = mulf %77, %cst_61 : vector<4xf64>
      %79 = addf %75, %78 : vector<4xf64>
      %80 = math.log %79 : vector<4xf64>
      %c0_62 = constant 0 : index
      vector.transfer_write %80, %arg1[%c0_62, %arg2] : vector<4xf64>, memref<1x?xf64>
    }
    %c1 = constant 1 : index
    scf.for %arg2 = %2 to %0 step %c1 {
      %3 = "lo_spn.batch_read"(%arg0, %arg2) {staticIndex = 0 : ui32} : (memref<?x6xf64>, index) -> f64
      %4 = "lo_spn.batch_read"(%arg0, %arg2) {staticIndex = 1 : ui32} : (memref<?x6xf64>, index) -> f64
      %5 = "lo_spn.batch_read"(%arg0, %arg2) {staticIndex = 2 : ui32} : (memref<?x6xf64>, index) -> f64
      %6 = "lo_spn.batch_read"(%arg0, %arg2) {staticIndex = 3 : ui32} : (memref<?x6xf64>, index) -> f64
      %7 = "lo_spn.batch_read"(%arg0, %arg2) {staticIndex = 4 : ui32} : (memref<?x6xf64>, index) -> f64
      %8 = "lo_spn.batch_read"(%arg0, %arg2) {staticIndex = 5 : ui32} : (memref<?x6xf64>, index) -> f64
      %9 = "lo_spn.categorical"(%3) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false} : (f64) -> f64
      %10 = "lo_spn.categorical"(%4) {probabilities = [2.500000e-01, 6.250000e-01, 1.250000e-01], supportMarginal = false} : (f64) -> f64
      %11 = "lo_spn.histogram"(%5) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}], supportMarginal = false} : (f64) -> f64
      %12 = "lo_spn.histogram"(%6) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 4.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 5.500000e-01 : f64}], supportMarginal = false} : (f64) -> f64
      %13 = "lo_spn.gaussian"(%7) {mean = 5.000000e-01 : f64, stddev = 1.000000e+00 : f64, supportMarginal = false} : (f64) -> f64
      %14 = "lo_spn.gaussian"(%8) {mean = 2.500000e-01 : f64, stddev = 1.000000e-01 : f64, supportMarginal = false} : (f64) -> f64
      %15 = "lo_spn.mul"(%9, %10) : (f64, f64) -> f64
      %16 = "lo_spn.mul"(%15, %11) : (f64, f64) -> f64
      %17 = "lo_spn.constant"() {type = f64, value = 1.000000e-01 : f64} : () -> f64
      %18 = "lo_spn.mul"(%16, %17) : (f64, f64) -> f64
      %19 = "lo_spn.mul"(%12, %13) : (f64, f64) -> f64
      %20 = "lo_spn.mul"(%19, %14) : (f64, f64) -> f64
      %21 = "lo_spn.constant"() {type = f64, value = 1.000000e-01 : f64} : () -> f64
      %22 = "lo_spn.mul"(%20, %21) : (f64, f64) -> f64
      %23 = "lo_spn.add"(%18, %22) : (f64, f64) -> f64
      %24 = "lo_spn.log"(%23) : (f64) -> f64
      "lo_spn.batch_write"(%arg1, %arg2, %24) {transposed = true} : (memref<1x?xf64>, index, f64) -> ()
    }
    return
  }
  func @spn_vector(%arg0: memref<?x6xf64>, %arg1: memref<1x?xf64>) {
    %c0 = constant 0 : index
    %0 = memref.dim %arg0, %c0 : memref<?x6xf64>
    %1 = memref.alloc(%0) : memref<1x?xf64>
    call @vec_task_0(%arg0, %1) : (memref<?x6xf64>, memref<1x?xf64>) -> ()
    %2 = memref.tensor_load %1 : memref<1x?xf64>
    %3 = memref.buffer_cast %2 : memref<1x?xf64>
    "lo_spn.copy"(%3, %arg1) : (memref<1x?xf64>, memref<1x?xf64>) -> ()
    "lo_spn.return"() : () -> ()
  }
}


// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

// CHECK-LABEL:   memref.global "private" constant @histogram_1 : memref<2xf64> = dense<[4.500000e-01, 5.500000e-01]>
// CHECK:         memref.global "private" constant @histogram_0 : memref<2xf64> = dense<[2.500000e-01, 7.500000e-01]>
// CHECK:         memref.global "private" constant @categorical_1 : memref<3xf64> = dense<[2.500000e-01, 6.250000e-01, 1.250000e-01]>
// CHECK:         memref.global "private" constant @categorical_0 : memref<3xf64> = dense<[3.500000e-01, 5.500000e-01, 1.000000e-01]>
// CHECK:         memref.global "private" constant @histogram_vec_1 : memref<2xf64> = dense<[4.500000e-01, 5.500000e-01]>
// CHECK:         memref.global "private" constant @histogram_vec_0 : memref<2xf64> = dense<[2.500000e-01, 7.500000e-01]>
// CHECK:         memref.global "private" constant @categorical_vec_1 : memref<3xf64> = dense<[2.500000e-01, 6.250000e-01, 1.250000e-01]>
// CHECK:         memref.global "private" constant @categorical_vec_0 : memref<3xf64> = dense<[3.500000e-01, 5.500000e-01, 1.000000e-01]>

// CHECK-LABEL:   func @vec_task_0(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<?x6xf64>,
// CHECK-SAME:                     %[[VAL_1:.*]]: memref<1x?xf64>) {
// CHECK:           %[[VAL_2:.*]] = constant 0 : index
// CHECK:           %[[VAL_3:.*]] = memref.dim %[[VAL_0]], %[[VAL_2]] : memref<?x6xf64>
// CHECK:           %[[VAL_4:.*]] = constant 4 : index
// CHECK:           %[[VAL_5:.*]] = remi_unsigned %[[VAL_3]], %[[VAL_4]] : index
// CHECK:           %[[VAL_6:.*]] = subi %[[VAL_3]], %[[VAL_5]] : index
// CHECK:           %[[VAL_7:.*]] = constant 0 : index
// CHECK:           %[[VAL_8:.*]] = constant 4 : index
// CHECK:           scf.for %[[VAL_9:.*]] = %[[VAL_7]] to %[[VAL_6]] step %[[VAL_8]] {
// CHECK:             %[[VAL_10:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_11:.*]] = vector.broadcast %[[VAL_10]] : i64 to vector<4xi64>
// CHECK:             %[[VAL_12:.*]] = constant dense<[0, 6, 12, 18]> : vector<4xi64>
// CHECK:             %[[VAL_13:.*]] = constant dense<6> : vector<4xi64>
// CHECK:             %[[VAL_14:.*]] = muli %[[VAL_11]], %[[VAL_13]] : vector<4xi64>
// CHECK:             %[[VAL_15:.*]] = addi %[[VAL_14]], %[[VAL_12]] : vector<4xi64>
// CHECK:             %[[VAL_16:.*]] = constant dense<0.000000e+00> : vector<4xf64>
// CHECK:             %[[VAL_17:.*]] = constant dense<true> : vector<4xi1>
// CHECK:             %[[VAL_18:.*]] = constant 0 : index
// CHECK:             %[[VAL_19:.*]] = memref.dim %[[VAL_0]], %[[VAL_18]] : memref<?x6xf64>
// CHECK:             %[[VAL_20:.*]] = constant 6 : index
// CHECK:             %[[VAL_21:.*]] = muli %[[VAL_19]], %[[VAL_20]] : index
// CHECK:             %[[VAL_22:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_21]]], strides: [1] : memref<?x6xf64> to memref<?xf64>
// CHECK:             %[[VAL_23:.*]] = constant 0 : index
// CHECK:             %[[VAL_24:.*]] = vector.gather %[[VAL_22]]{{\[}}%[[VAL_23]]] {{\[}}%[[VAL_15]]], %[[VAL_17]], %[[VAL_16]] : memref<?xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
// CHECK:             %[[VAL_25:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_26:.*]] = vector.broadcast %[[VAL_25]] : i64 to vector<4xi64>
// CHECK:             %[[VAL_27:.*]] = constant dense<[1, 7, 13, 19]> : vector<4xi64>
// CHECK:             %[[VAL_28:.*]] = constant dense<6> : vector<4xi64>
// CHECK:             %[[VAL_29:.*]] = muli %[[VAL_26]], %[[VAL_28]] : vector<4xi64>
// CHECK:             %[[VAL_30:.*]] = addi %[[VAL_29]], %[[VAL_27]] : vector<4xi64>
// CHECK:             %[[VAL_31:.*]] = constant dense<0.000000e+00> : vector<4xf64>
// CHECK:             %[[VAL_32:.*]] = constant dense<true> : vector<4xi1>
// CHECK:             %[[VAL_33:.*]] = constant 0 : index
// CHECK:             %[[VAL_34:.*]] = memref.dim %[[VAL_0]], %[[VAL_33]] : memref<?x6xf64>
// CHECK:             %[[VAL_35:.*]] = constant 6 : index
// CHECK:             %[[VAL_36:.*]] = muli %[[VAL_34]], %[[VAL_35]] : index
// CHECK:             %[[VAL_37:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_36]]], strides: [1] : memref<?x6xf64> to memref<?xf64>
// CHECK:             %[[VAL_38:.*]] = constant 0 : index
// CHECK:             %[[VAL_39:.*]] = vector.gather %[[VAL_37]]{{\[}}%[[VAL_38]]] {{\[}}%[[VAL_30]]], %[[VAL_32]], %[[VAL_31]] : memref<?xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
// CHECK:             %[[VAL_40:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_41:.*]] = vector.broadcast %[[VAL_40]] : i64 to vector<4xi64>
// CHECK:             %[[VAL_42:.*]] = constant dense<[2, 8, 14, 20]> : vector<4xi64>
// CHECK:             %[[VAL_43:.*]] = constant dense<6> : vector<4xi64>
// CHECK:             %[[VAL_44:.*]] = muli %[[VAL_41]], %[[VAL_43]] : vector<4xi64>
// CHECK:             %[[VAL_45:.*]] = addi %[[VAL_44]], %[[VAL_42]] : vector<4xi64>
// CHECK:             %[[VAL_46:.*]] = constant dense<0.000000e+00> : vector<4xf64>
// CHECK:             %[[VAL_47:.*]] = constant dense<true> : vector<4xi1>
// CHECK:             %[[VAL_48:.*]] = constant 0 : index
// CHECK:             %[[VAL_49:.*]] = memref.dim %[[VAL_0]], %[[VAL_48]] : memref<?x6xf64>
// CHECK:             %[[VAL_50:.*]] = constant 6 : index
// CHECK:             %[[VAL_51:.*]] = muli %[[VAL_49]], %[[VAL_50]] : index
// CHECK:             %[[VAL_52:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_51]]], strides: [1] : memref<?x6xf64> to memref<?xf64>
// CHECK:             %[[VAL_53:.*]] = constant 0 : index
// CHECK:             %[[VAL_54:.*]] = vector.gather %[[VAL_52]]{{\[}}%[[VAL_53]]] {{\[}}%[[VAL_45]]], %[[VAL_47]], %[[VAL_46]] : memref<?xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
// CHECK:             %[[VAL_55:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_56:.*]] = vector.broadcast %[[VAL_55]] : i64 to vector<4xi64>
// CHECK:             %[[VAL_57:.*]] = constant dense<[3, 9, 15, 21]> : vector<4xi64>
// CHECK:             %[[VAL_58:.*]] = constant dense<6> : vector<4xi64>
// CHECK:             %[[VAL_59:.*]] = muli %[[VAL_56]], %[[VAL_58]] : vector<4xi64>
// CHECK:             %[[VAL_60:.*]] = addi %[[VAL_59]], %[[VAL_57]] : vector<4xi64>
// CHECK:             %[[VAL_61:.*]] = constant dense<0.000000e+00> : vector<4xf64>
// CHECK:             %[[VAL_62:.*]] = constant dense<true> : vector<4xi1>
// CHECK:             %[[VAL_63:.*]] = constant 0 : index
// CHECK:             %[[VAL_64:.*]] = memref.dim %[[VAL_0]], %[[VAL_63]] : memref<?x6xf64>
// CHECK:             %[[VAL_65:.*]] = constant 6 : index
// CHECK:             %[[VAL_66:.*]] = muli %[[VAL_64]], %[[VAL_65]] : index
// CHECK:             %[[VAL_67:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_66]]], strides: [1] : memref<?x6xf64> to memref<?xf64>
// CHECK:             %[[VAL_68:.*]] = constant 0 : index
// CHECK:             %[[VAL_69:.*]] = vector.gather %[[VAL_67]]{{\[}}%[[VAL_68]]] {{\[}}%[[VAL_60]]], %[[VAL_62]], %[[VAL_61]] : memref<?xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
// CHECK:             %[[VAL_70:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_71:.*]] = vector.broadcast %[[VAL_70]] : i64 to vector<4xi64>
// CHECK:             %[[VAL_72:.*]] = constant dense<[4, 10, 16, 22]> : vector<4xi64>
// CHECK:             %[[VAL_73:.*]] = constant dense<6> : vector<4xi64>
// CHECK:             %[[VAL_74:.*]] = muli %[[VAL_71]], %[[VAL_73]] : vector<4xi64>
// CHECK:             %[[VAL_75:.*]] = addi %[[VAL_74]], %[[VAL_72]] : vector<4xi64>
// CHECK:             %[[VAL_76:.*]] = constant dense<0.000000e+00> : vector<4xf64>
// CHECK:             %[[VAL_77:.*]] = constant dense<true> : vector<4xi1>
// CHECK:             %[[VAL_78:.*]] = constant 0 : index
// CHECK:             %[[VAL_79:.*]] = memref.dim %[[VAL_0]], %[[VAL_78]] : memref<?x6xf64>
// CHECK:             %[[VAL_80:.*]] = constant 6 : index
// CHECK:             %[[VAL_81:.*]] = muli %[[VAL_79]], %[[VAL_80]] : index
// CHECK:             %[[VAL_82:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_81]]], strides: [1] : memref<?x6xf64> to memref<?xf64>
// CHECK:             %[[VAL_83:.*]] = constant 0 : index
// CHECK:             %[[VAL_84:.*]] = vector.gather %[[VAL_82]]{{\[}}%[[VAL_83]]] {{\[}}%[[VAL_75]]], %[[VAL_77]], %[[VAL_76]] : memref<?xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
// CHECK:             %[[VAL_85:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_86:.*]] = vector.broadcast %[[VAL_85]] : i64 to vector<4xi64>
// CHECK:             %[[VAL_87:.*]] = constant dense<[5, 11, 17, 23]> : vector<4xi64>
// CHECK:             %[[VAL_88:.*]] = constant dense<6> : vector<4xi64>
// CHECK:             %[[VAL_89:.*]] = muli %[[VAL_86]], %[[VAL_88]] : vector<4xi64>
// CHECK:             %[[VAL_90:.*]] = addi %[[VAL_89]], %[[VAL_87]] : vector<4xi64>
// CHECK:             %[[VAL_91:.*]] = constant dense<0.000000e+00> : vector<4xf64>
// CHECK:             %[[VAL_92:.*]] = constant dense<true> : vector<4xi1>
// CHECK:             %[[VAL_93:.*]] = constant 0 : index
// CHECK:             %[[VAL_94:.*]] = memref.dim %[[VAL_0]], %[[VAL_93]] : memref<?x6xf64>
// CHECK:             %[[VAL_95:.*]] = constant 6 : index
// CHECK:             %[[VAL_96:.*]] = muli %[[VAL_94]], %[[VAL_95]] : index
// CHECK:             %[[VAL_97:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_96]]], strides: [1] : memref<?x6xf64> to memref<?xf64>
// CHECK:             %[[VAL_98:.*]] = constant 0 : index
// CHECK:             %[[VAL_99:.*]] = vector.gather %[[VAL_97]]{{\[}}%[[VAL_98]]] {{\[}}%[[VAL_90]]], %[[VAL_92]], %[[VAL_91]] : memref<?xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
// CHECK:             %[[VAL_100:.*]] = memref.get_global @categorical_vec_0 : memref<3xf64>
// CHECK:             %[[VAL_101:.*]] = fptoui %[[VAL_24]] : vector<4xf64> to vector<4xi64>
// CHECK:             %[[VAL_102:.*]] = constant dense<0.000000e+00> : vector<4xf64>
// CHECK:             %[[VAL_103:.*]] = constant dense<true> : vector<4xi1>
// CHECK:             %[[VAL_104:.*]] = constant 0 : index
// CHECK:             %[[VAL_105:.*]] = vector.gather %[[VAL_100]]{{\[}}%[[VAL_104]]] {{\[}}%[[VAL_101]]], %[[VAL_103]], %[[VAL_102]] : memref<3xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
// CHECK:             %[[VAL_106:.*]] = memref.get_global @categorical_vec_1 : memref<3xf64>
// CHECK:             %[[VAL_107:.*]] = fptoui %[[VAL_39]] : vector<4xf64> to vector<4xi64>
// CHECK:             %[[VAL_108:.*]] = constant dense<0.000000e+00> : vector<4xf64>
// CHECK:             %[[VAL_109:.*]] = constant dense<true> : vector<4xi1>
// CHECK:             %[[VAL_110:.*]] = constant 0 : index
// CHECK:             %[[VAL_111:.*]] = vector.gather %[[VAL_106]]{{\[}}%[[VAL_110]]] {{\[}}%[[VAL_107]]], %[[VAL_109]], %[[VAL_108]] : memref<3xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
// CHECK:             %[[VAL_112:.*]] = memref.get_global @histogram_vec_0 : memref<2xf64>
// CHECK:             %[[VAL_113:.*]] = fptoui %[[VAL_54]] : vector<4xf64> to vector<4xi64>
// CHECK:             %[[VAL_114:.*]] = constant dense<0.000000e+00> : vector<4xf64>
// CHECK:             %[[VAL_115:.*]] = constant dense<true> : vector<4xi1>
// CHECK:             %[[VAL_116:.*]] = constant 0 : index
// CHECK:             %[[VAL_117:.*]] = vector.gather %[[VAL_112]]{{\[}}%[[VAL_116]]] {{\[}}%[[VAL_113]]], %[[VAL_115]], %[[VAL_114]] : memref<2xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
// CHECK:             %[[VAL_118:.*]] = memref.get_global @histogram_vec_1 : memref<2xf64>
// CHECK:             %[[VAL_119:.*]] = fptoui %[[VAL_69]] : vector<4xf64> to vector<4xi64>
// CHECK:             %[[VAL_120:.*]] = constant dense<0.000000e+00> : vector<4xf64>
// CHECK:             %[[VAL_121:.*]] = constant dense<true> : vector<4xi1>
// CHECK:             %[[VAL_122:.*]] = constant 0 : index
// CHECK:             %[[VAL_123:.*]] = vector.gather %[[VAL_118]]{{\[}}%[[VAL_122]]] {{\[}}%[[VAL_119]]], %[[VAL_121]], %[[VAL_120]] : memref<2xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
// CHECK:             %[[VAL_124:.*]] = constant dense<0.3989422804014327> : vector<4xf64>
// CHECK:             %[[VAL_125:.*]] = constant dense<-5.000000e-01> : vector<4xf64>
// CHECK:             %[[VAL_126:.*]] = constant dense<5.000000e-01> : vector<4xf64>
// CHECK:             %[[VAL_127:.*]] = subf %[[VAL_84]], %[[VAL_126]] : vector<4xf64>
// CHECK:             %[[VAL_128:.*]] = mulf %[[VAL_127]], %[[VAL_127]] : vector<4xf64>
// CHECK:             %[[VAL_129:.*]] = mulf %[[VAL_128]], %[[VAL_125]] : vector<4xf64>
// CHECK:             %[[VAL_130:.*]] = math.exp %[[VAL_129]] : vector<4xf64>
// CHECK:             %[[VAL_131:.*]] = mulf %[[VAL_124]], %[[VAL_130]] : vector<4xf64>
// CHECK:             %[[VAL_132:.*]] = constant dense<3.9894228040143269> : vector<4xf64>
// CHECK:             %[[VAL_133:.*]] = constant dense<-49.999999999999993> : vector<4xf64>
// CHECK:             %[[VAL_134:.*]] = constant dense<2.500000e-01> : vector<4xf64>
// CHECK:             %[[VAL_135:.*]] = subf %[[VAL_99]], %[[VAL_134]] : vector<4xf64>
// CHECK:             %[[VAL_136:.*]] = mulf %[[VAL_135]], %[[VAL_135]] : vector<4xf64>
// CHECK:             %[[VAL_137:.*]] = mulf %[[VAL_136]], %[[VAL_133]] : vector<4xf64>
// CHECK:             %[[VAL_138:.*]] = math.exp %[[VAL_137]] : vector<4xf64>
// CHECK:             %[[VAL_139:.*]] = mulf %[[VAL_132]], %[[VAL_138]] : vector<4xf64>
// CHECK:             %[[VAL_140:.*]] = mulf %[[VAL_105]], %[[VAL_111]] : vector<4xf64>
// CHECK:             %[[VAL_141:.*]] = mulf %[[VAL_140]], %[[VAL_117]] : vector<4xf64>
// CHECK:             %[[VAL_142:.*]] = constant dense<1.000000e-01> : vector<4xf64>
// CHECK:             %[[VAL_143:.*]] = mulf %[[VAL_141]], %[[VAL_142]] : vector<4xf64>
// CHECK:             %[[VAL_144:.*]] = mulf %[[VAL_123]], %[[VAL_131]] : vector<4xf64>
// CHECK:             %[[VAL_145:.*]] = mulf %[[VAL_144]], %[[VAL_139]] : vector<4xf64>
// CHECK:             %[[VAL_146:.*]] = constant dense<1.000000e-01> : vector<4xf64>
// CHECK:             %[[VAL_147:.*]] = mulf %[[VAL_145]], %[[VAL_146]] : vector<4xf64>
// CHECK:             %[[VAL_148:.*]] = addf %[[VAL_143]], %[[VAL_147]] : vector<4xf64>
// CHECK:             %[[VAL_149:.*]] = math.log %[[VAL_148]] : vector<4xf64>
// CHECK:             %[[VAL_150:.*]] = constant 0 : index
// CHECK:             vector.transfer_write %[[VAL_149]], %[[VAL_1]]{{\[}}%[[VAL_150]], %[[VAL_9]]] : vector<4xf64>, memref<1x?xf64>
// CHECK:           }
// CHECK:           %[[VAL_151:.*]] = constant 1 : index
// CHECK:           scf.for %[[VAL_152:.*]] = %[[VAL_6]] to %[[VAL_3]] step %[[VAL_151]] {
// CHECK:             %[[VAL_153:.*]] = constant 0 : index
// CHECK:             %[[VAL_154:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_152]], %[[VAL_153]]] : memref<?x6xf64>
// CHECK:             %[[VAL_155:.*]] = constant 1 : index
// CHECK:             %[[VAL_156:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_152]], %[[VAL_155]]] : memref<?x6xf64>
// CHECK:             %[[VAL_157:.*]] = constant 2 : index
// CHECK:             %[[VAL_158:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_152]], %[[VAL_157]]] : memref<?x6xf64>
// CHECK:             %[[VAL_159:.*]] = constant 3 : index
// CHECK:             %[[VAL_160:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_152]], %[[VAL_159]]] : memref<?x6xf64>
// CHECK:             %[[VAL_161:.*]] = constant 4 : index
// CHECK:             %[[VAL_162:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_152]], %[[VAL_161]]] : memref<?x6xf64>
// CHECK:             %[[VAL_163:.*]] = constant 5 : index
// CHECK:             %[[VAL_164:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_152]], %[[VAL_163]]] : memref<?x6xf64>
// CHECK:             %[[VAL_165:.*]] = memref.get_global @categorical_0 : memref<3xf64>
// CHECK:             %[[VAL_166:.*]] = fptoui %[[VAL_154]] : f64 to i64
// CHECK:             %[[VAL_167:.*]] = index_cast %[[VAL_166]] : i64 to index
// CHECK:             %[[VAL_168:.*]] = memref.load %[[VAL_165]]{{\[}}%[[VAL_167]]] : memref<3xf64>
// CHECK:             %[[VAL_169:.*]] = memref.get_global @categorical_1 : memref<3xf64>
// CHECK:             %[[VAL_170:.*]] = fptoui %[[VAL_156]] : f64 to i64
// CHECK:             %[[VAL_171:.*]] = index_cast %[[VAL_170]] : i64 to index
// CHECK:             %[[VAL_172:.*]] = memref.load %[[VAL_169]]{{\[}}%[[VAL_171]]] : memref<3xf64>
// CHECK:             %[[VAL_173:.*]] = memref.get_global @histogram_0 : memref<2xf64>
// CHECK:             %[[VAL_174:.*]] = fptoui %[[VAL_158]] : f64 to i64
// CHECK:             %[[VAL_175:.*]] = index_cast %[[VAL_174]] : i64 to index
// CHECK:             %[[VAL_176:.*]] = memref.load %[[VAL_173]]{{\[}}%[[VAL_175]]] : memref<2xf64>
// CHECK:             %[[VAL_177:.*]] = memref.get_global @histogram_1 : memref<2xf64>
// CHECK:             %[[VAL_178:.*]] = fptoui %[[VAL_160]] : f64 to i64
// CHECK:             %[[VAL_179:.*]] = index_cast %[[VAL_178]] : i64 to index
// CHECK:             %[[VAL_180:.*]] = memref.load %[[VAL_177]]{{\[}}%[[VAL_179]]] : memref<2xf64>
// CHECK:             %[[VAL_181:.*]] = constant 0.3989422804014327 : f64
// CHECK:             %[[VAL_182:.*]] = constant -5.000000e-01 : f64
// CHECK:             %[[VAL_183:.*]] = constant 5.000000e-01 : f64
// CHECK:             %[[VAL_184:.*]] = subf %[[VAL_162]], %[[VAL_183]] : f64
// CHECK:             %[[VAL_185:.*]] = mulf %[[VAL_184]], %[[VAL_184]] : f64
// CHECK:             %[[VAL_186:.*]] = mulf %[[VAL_185]], %[[VAL_182]] : f64
// CHECK:             %[[VAL_187:.*]] = math.exp %[[VAL_186]] : f64
// CHECK:             %[[VAL_188:.*]] = mulf %[[VAL_181]], %[[VAL_187]] : f64
// CHECK:             %[[VAL_189:.*]] = constant 3.9894228040143269 : f64
// CHECK:             %[[VAL_190:.*]] = constant -49.999999999999993 : f64
// CHECK:             %[[VAL_191:.*]] = constant 2.500000e-01 : f64
// CHECK:             %[[VAL_192:.*]] = subf %[[VAL_164]], %[[VAL_191]] : f64
// CHECK:             %[[VAL_193:.*]] = mulf %[[VAL_192]], %[[VAL_192]] : f64
// CHECK:             %[[VAL_194:.*]] = mulf %[[VAL_193]], %[[VAL_190]] : f64
// CHECK:             %[[VAL_195:.*]] = math.exp %[[VAL_194]] : f64
// CHECK:             %[[VAL_196:.*]] = mulf %[[VAL_189]], %[[VAL_195]] : f64
// CHECK:             %[[VAL_197:.*]] = mulf %[[VAL_168]], %[[VAL_172]] : f64
// CHECK:             %[[VAL_198:.*]] = mulf %[[VAL_197]], %[[VAL_176]] : f64
// CHECK:             %[[VAL_199:.*]] = constant 1.000000e-01 : f64
// CHECK:             %[[VAL_200:.*]] = mulf %[[VAL_198]], %[[VAL_199]] : f64
// CHECK:             %[[VAL_201:.*]] = mulf %[[VAL_180]], %[[VAL_188]] : f64
// CHECK:             %[[VAL_202:.*]] = mulf %[[VAL_201]], %[[VAL_196]] : f64
// CHECK:             %[[VAL_203:.*]] = constant 1.000000e-01 : f64
// CHECK:             %[[VAL_204:.*]] = mulf %[[VAL_202]], %[[VAL_203]] : f64
// CHECK:             %[[VAL_205:.*]] = addf %[[VAL_200]], %[[VAL_204]] : f64
// CHECK:             %[[VAL_206:.*]] = math.log %[[VAL_205]] : f64
// CHECK:             %[[VAL_207:.*]] = constant 0 : index
// CHECK:             memref.store %[[VAL_206]], %[[VAL_1]]{{\[}}%[[VAL_207]], %[[VAL_152]]] : memref<1x?xf64>
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func @spn_vector(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<?x6xf64>,
// CHECK-SAME:                     %[[VAL_1:.*]]: memref<1x?xf64>) {
// CHECK:           %[[VAL_2:.*]] = constant 0 : index
// CHECK:           %[[VAL_3:.*]] = memref.dim %[[VAL_0]], %[[VAL_2]] : memref<?x6xf64>
// CHECK:           %[[VAL_4:.*]] = memref.alloc(%[[VAL_3]]) : memref<1x?xf64>
// CHECK:           call @vec_task_0(%[[VAL_0]], %[[VAL_4]]) : (memref<?x6xf64>, memref<1x?xf64>) -> ()
// CHECK:           %[[VAL_5:.*]] = memref.tensor_load %[[VAL_4]] : memref<1x?xf64>
// CHECK:           %[[VAL_6:.*]] = memref.buffer_cast %[[VAL_5]] : memref<1x?xf64>
// CHECK:           %[[VAL_7:.*]] = constant 1 : index
// CHECK:           %[[VAL_8:.*]] = memref.dim %[[VAL_6]], %[[VAL_7]] : memref<1x?xf64>
// CHECK:           %[[VAL_9:.*]] = constant 0 : index
// CHECK:           %[[VAL_10:.*]] = constant 1 : index
// CHECK:           scf.for %[[VAL_11:.*]] = %[[VAL_9]] to %[[VAL_8]] step %[[VAL_10]] {
// CHECK:             %[[VAL_12:.*]] = constant 0 : index
// CHECK:             %[[VAL_13:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_12]], %[[VAL_11]]] : memref<1x?xf64>
// CHECK:             memref.store %[[VAL_13]], %[[VAL_1]]{{\[}}%[[VAL_12]], %[[VAL_11]]] : memref<1x?xf64>
// CHECK:           }
// CHECK:           return
// CHECK:         }
