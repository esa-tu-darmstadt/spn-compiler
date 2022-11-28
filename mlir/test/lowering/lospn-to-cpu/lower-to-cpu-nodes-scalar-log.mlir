// RUN: %optcall --convert-lospn-nodes-to-cpu %s | FileCheck %s

module  {
  memref.global "private" arith.constant @histogram_vec_1 : memref<2xf32> = dense<[-0.79850769, -0.597836971]>
  memref.global "private" arith.constant @histogram_vec_0 : memref<2xf32> = dense<[-1.38629436, -0.287682086]>
  memref.global "private" arith.constant @categorical_vec_1 : memref<3xf32> = dense<[-1.38629436, -0.470003635, -2.07944155]>
  memref.global "private" arith.constant @categorical_vec_0 : memref<3xf32> = dense<[-1.04982209, -0.597836971, -2.30258512]>
  func.func @vec_task_0(%arg0: memref<?x6xf32>, %arg1: memref<1x?xf32>) {
    %c0 = arith.constant 0 : index
    %0 = memref.dim %arg0, %c0 : memref<?x6xf32>
    %c4 = arith.constant 4 : index
    %1 = remi_unsigned %0, %c4 : index
    %2 = subi %0, %1 : index
    %c0_0 = arith.constant 0 : index
    %c4_1 = arith.constant 4 : index
    scf.for %arg2 = %c0_0 to %2 step %c4_1 {
      %3 = index_cast %arg2 : index to i64
      %4 = vector.broadcast %3 : i64 to vector<8xi64>
      %cst = arith.constant dense<[0, 6, 12, 18, 24, 30, 36, 42]> : vector<8xi64>
      %cst_2 = arith.constant dense<6> : vector<8xi64>
      %5 = muli %4, %cst_2 : vector<8xi64>
      %6 = addi %5, %cst : vector<8xi64>
      %cst_3 = arith.constant dense<0.000000e+00> : vector<8xf32>
      %cst_4 = arith.constant dense<true> : vector<8xi1>
      %c0_5 = arith.constant 0 : index
      %7 = memref.dim %arg0, %c0_5 : memref<?x6xf32>
      %c6 = arith.constant 6 : index
      %8 = muli %7, %c6 : index
      %9 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%8], strides: [1] : memref<?x6xf32> to memref<?xf32>
      %c0_6 = arith.constant 0 : index
      %10 = vector.gather %9[%c0_6] [%6], %cst_4, %cst_3 : memref<?xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
      %11 = index_cast %arg2 : index to i64
      %12 = vector.broadcast %11 : i64 to vector<8xi64>
      %cst_7 = arith.constant dense<[1, 7, 13, 19, 25, 31, 37, 43]> : vector<8xi64>
      %cst_8 = arith.constant dense<6> : vector<8xi64>
      %13 = muli %12, %cst_8 : vector<8xi64>
      %14 = addi %13, %cst_7 : vector<8xi64>
      %cst_9 = arith.constant dense<0.000000e+00> : vector<8xf32>
      %cst_10 = arith.constant dense<true> : vector<8xi1>
      %c0_11 = arith.constant 0 : index
      %15 = memref.dim %arg0, %c0_11 : memref<?x6xf32>
      %c6_12 = arith.constant 6 : index
      %16 = muli %15, %c6_12 : index
      %17 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%16], strides: [1] : memref<?x6xf32> to memref<?xf32>
      %c0_13 = arith.constant 0 : index
      %18 = vector.gather %17[%c0_13] [%14], %cst_10, %cst_9 : memref<?xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
      %19 = index_cast %arg2 : index to i64
      %20 = vector.broadcast %19 : i64 to vector<8xi64>
      %cst_14 = arith.constant dense<[2, 8, 14, 20, 26, 32, 38, 44]> : vector<8xi64>
      %cst_15 = arith.constant dense<6> : vector<8xi64>
      %21 = muli %20, %cst_15 : vector<8xi64>
      %22 = addi %21, %cst_14 : vector<8xi64>
      %cst_16 = arith.constant dense<0.000000e+00> : vector<8xf32>
      %cst_17 = arith.constant dense<true> : vector<8xi1>
      %c0_18 = arith.constant 0 : index
      %23 = memref.dim %arg0, %c0_18 : memref<?x6xf32>
      %c6_19 = arith.constant 6 : index
      %24 = muli %23, %c6_19 : index
      %25 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%24], strides: [1] : memref<?x6xf32> to memref<?xf32>
      %c0_20 = arith.constant 0 : index
      %26 = vector.gather %25[%c0_20] [%22], %cst_17, %cst_16 : memref<?xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
      %27 = index_cast %arg2 : index to i64
      %28 = vector.broadcast %27 : i64 to vector<8xi64>
      %cst_21 = arith.constant dense<[3, 9, 15, 21, 27, 33, 39, 45]> : vector<8xi64>
      %cst_22 = arith.constant dense<6> : vector<8xi64>
      %29 = muli %28, %cst_22 : vector<8xi64>
      %30 = addi %29, %cst_21 : vector<8xi64>
      %cst_23 = arith.constant dense<0.000000e+00> : vector<8xf32>
      %cst_24 = arith.constant dense<true> : vector<8xi1>
      %c0_25 = arith.constant 0 : index
      %31 = memref.dim %arg0, %c0_25 : memref<?x6xf32>
      %c6_26 = arith.constant 6 : index
      %32 = muli %31, %c6_26 : index
      %33 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%32], strides: [1] : memref<?x6xf32> to memref<?xf32>
      %c0_27 = arith.constant 0 : index
      %34 = vector.gather %33[%c0_27] [%30], %cst_24, %cst_23 : memref<?xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
      %35 = index_cast %arg2 : index to i64
      %36 = vector.broadcast %35 : i64 to vector<8xi64>
      %cst_28 = arith.constant dense<[4, 10, 16, 22, 28, 34, 40, 46]> : vector<8xi64>
      %cst_29 = arith.constant dense<6> : vector<8xi64>
      %37 = muli %36, %cst_29 : vector<8xi64>
      %38 = addi %37, %cst_28 : vector<8xi64>
      %cst_30 = arith.constant dense<0.000000e+00> : vector<8xf32>
      %cst_31 = arith.constant dense<true> : vector<8xi1>
      %c0_32 = arith.constant 0 : index
      %39 = memref.dim %arg0, %c0_32 : memref<?x6xf32>
      %c6_33 = arith.constant 6 : index
      %40 = muli %39, %c6_33 : index
      %41 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%40], strides: [1] : memref<?x6xf32> to memref<?xf32>
      %c0_34 = arith.constant 0 : index
      %42 = vector.gather %41[%c0_34] [%38], %cst_31, %cst_30 : memref<?xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
      %43 = index_cast %arg2 : index to i64
      %44 = vector.broadcast %43 : i64 to vector<8xi64>
      %cst_35 = arith.constant dense<[5, 11, 17, 23, 29, 35, 41, 47]> : vector<8xi64>
      %cst_36 = arith.constant dense<6> : vector<8xi64>
      %45 = muli %44, %cst_36 : vector<8xi64>
      %46 = addi %45, %cst_35 : vector<8xi64>
      %cst_37 = arith.constant dense<0.000000e+00> : vector<8xf32>
      %cst_38 = arith.constant dense<true> : vector<8xi1>
      %c0_39 = arith.constant 0 : index
      %47 = memref.dim %arg0, %c0_39 : memref<?x6xf32>
      %c6_40 = arith.constant 6 : index
      %48 = muli %47, %c6_40 : index
      %49 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%48], strides: [1] : memref<?x6xf32> to memref<?xf32>
      %c0_41 = arith.constant 0 : index
      %50 = vector.gather %49[%c0_41] [%46], %cst_38, %cst_37 : memref<?xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
      %51 = memref.get_global @categorical_vec_0 : memref<3xf32>
      %52 = fptoui %10 : vector<8xf32> to vector<8xi64>
      %cst_42 = arith.constant dense<0.000000e+00> : vector<8xf32>
      %cst_43 = arith.constant dense<true> : vector<8xi1>
      %c0_44 = arith.constant 0 : index
      %53 = vector.gather %51[%c0_44] [%52], %cst_43, %cst_42 : memref<3xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
      %54 = memref.get_global @categorical_vec_1 : memref<3xf32>
      %55 = fptoui %18 : vector<8xf32> to vector<8xi64>
      %cst_45 = arith.constant dense<0.000000e+00> : vector<8xf32>
      %cst_46 = arith.constant dense<true> : vector<8xi1>
      %c0_47 = arith.constant 0 : index
      %56 = vector.gather %54[%c0_47] [%55], %cst_46, %cst_45 : memref<3xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
      %57 = memref.get_global @histogram_vec_0 : memref<2xf32>
      %58 = fptoui %26 : vector<8xf32> to vector<8xi64>
      %cst_48 = arith.constant dense<0.000000e+00> : vector<8xf32>
      %cst_49 = arith.constant dense<true> : vector<8xi1>
      %c0_50 = arith.constant 0 : index
      %59 = vector.gather %57[%c0_50] [%58], %cst_49, %cst_48 : memref<2xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
      %60 = memref.get_global @histogram_vec_1 : memref<2xf32>
      %61 = fptoui %34 : vector<8xf32> to vector<8xi64>
      %cst_51 = arith.constant dense<0.000000e+00> : vector<8xf32>
      %cst_52 = arith.constant dense<true> : vector<8xi1>
      %c0_53 = arith.constant 0 : index
      %62 = vector.gather %60[%c0_53] [%61], %cst_52, %cst_51 : memref<2xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
      %cst_54 = arith.constant dense<-5.000000e-01> : vector<8xf32>
      %cst_55 = arith.constant dense<-0.918938517> : vector<8xf32>
      %cst_56 = arith.constant dense<5.000000e-01> : vector<8xf32>
      %63 = subf %42, %cst_56 : vector<8xf32>
      %64 = mulf %63, %63 : vector<8xf32>
      %65 = mulf %64, %cst_54 : vector<8xf32>
      %66 = addf %cst_55, %65 : vector<8xf32>
      %cst_57 = arith.constant dense<-5.000000e+01> : vector<8xf32>
      %cst_58 = arith.constant dense<1.38364661> : vector<8xf32>
      %cst_59 = arith.constant dense<2.500000e-01> : vector<8xf32>
      %67 = subf %50, %cst_59 : vector<8xf32>
      %68 = mulf %67, %67 : vector<8xf32>
      %69 = mulf %68, %cst_57 : vector<8xf32>
      %70 = addf %cst_58, %69 : vector<8xf32>
      %71 = addf %53, %56 : vector<8xf32>
      %72 = addf %71, %59 : vector<8xf32>
      %cst_60 = arith.constant dense<1.000000e-01> : vector<8xf32>
      %73 = addf %72, %cst_60 : vector<8xf32>
      %74 = addf %62, %66 : vector<8xf32>
      %75 = addf %74, %70 : vector<8xf32>
      %cst_61 = arith.constant dense<1.000000e-01> : vector<8xf32>
      %76 = addf %75, %cst_61 : vector<8xf32>
      %77 = cmpf ogt, %73, %76 : vector<8xf32>
      %78 = select %77, %73, %76 : vector<8xi1>, vector<8xf32>
      %79 = select %77, %76, %73 : vector<8xi1>, vector<8xf32>
      %80 = subf %79, %78 : vector<8xf32>
      %81 = math.exp %80 : vector<8xf32>
      %82 = math.log1p %81 : vector<8xf32>
      %83 = addf %78, %82 : vector<8xf32>
      %c0_62 = arith.constant 0 : index
      vector.transfer_write %83, %arg1[%c0_62, %arg2] : vector<8xf32>, memref<1x?xf32>
    }
    %c1 = arith.constant 1 : index
    scf.for %arg2 = %2 to %0 step %c1 {
      %3 = "lo_spn.batch_read"(%arg0, %arg2) {staticIndex = 0 : ui32} : (memref<?x6xf32>, index) -> f32
      %4 = "lo_spn.batch_read"(%arg0, %arg2) {staticIndex = 1 : ui32} : (memref<?x6xf32>, index) -> f32
      %5 = "lo_spn.batch_read"(%arg0, %arg2) {staticIndex = 2 : ui32} : (memref<?x6xf32>, index) -> f32
      %6 = "lo_spn.batch_read"(%arg0, %arg2) {staticIndex = 3 : ui32} : (memref<?x6xf32>, index) -> f32
      %7 = "lo_spn.batch_read"(%arg0, %arg2) {staticIndex = 4 : ui32} : (memref<?x6xf32>, index) -> f32
      %8 = "lo_spn.batch_read"(%arg0, %arg2) {staticIndex = 5 : ui32} : (memref<?x6xf32>, index) -> f32
      %9 = "lo_spn.categorical"(%3) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false} : (f32) -> !lo_spn.log<f32>
      %10 = "lo_spn.categorical"(%4) {probabilities = [2.500000e-01, 6.250000e-01, 1.250000e-01], supportMarginal = false} : (f32) -> !lo_spn.log<f32>
      %11 = "lo_spn.histogram"(%5) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}], supportMarginal = false} : (f32) -> !lo_spn.log<f32>
      %12 = "lo_spn.histogram"(%6) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 4.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 5.500000e-01 : f64}], supportMarginal = false} : (f32) -> !lo_spn.log<f32>
      %13 = "lo_spn.gaussian"(%7) {mean = 5.000000e-01 : f64, stddev = 1.000000e+00 : f64, supportMarginal = false} : (f32) -> !lo_spn.log<f32>
      %14 = "lo_spn.gaussian"(%8) {mean = 2.500000e-01 : f64, stddev = 1.000000e-01 : f64, supportMarginal = false} : (f32) -> !lo_spn.log<f32>
      %15 = "lo_spn.mul"(%9, %10) : (!lo_spn.log<f32>, !lo_spn.log<f32>) -> !lo_spn.log<f32>
      %16 = "lo_spn.mul"(%15, %11) : (!lo_spn.log<f32>, !lo_spn.log<f32>) -> !lo_spn.log<f32>
      %17 = "lo_spn.arith.constant"() {type = !lo_spn.log<f32>, value = 1.000000e-01 : f64} : () -> !lo_spn.log<f32>
      %18 = "lo_spn.mul"(%16, %17) : (!lo_spn.log<f32>, !lo_spn.log<f32>) -> !lo_spn.log<f32>
      %19 = "lo_spn.mul"(%12, %13) : (!lo_spn.log<f32>, !lo_spn.log<f32>) -> !lo_spn.log<f32>
      %20 = "lo_spn.mul"(%19, %14) : (!lo_spn.log<f32>, !lo_spn.log<f32>) -> !lo_spn.log<f32>
      %21 = "lo_spn.arith.constant"() {type = !lo_spn.log<f32>, value = 1.000000e-01 : f64} : () -> !lo_spn.log<f32>
      %22 = "lo_spn.mul"(%20, %21) : (!lo_spn.log<f32>, !lo_spn.log<f32>) -> !lo_spn.log<f32>
      %23 = "lo_spn.add"(%18, %22) : (!lo_spn.log<f32>, !lo_spn.log<f32>) -> !lo_spn.log<f32>
      %24 = "lo_spn.strip_log"(%23) {target = f32} : (!lo_spn.log<f32>) -> f32
      "lo_spn.batch_write"(%arg1, %arg2, %24) {transposed = true} : (memref<1x?xf32>, index, f32) -> ()
    }
    return
  }
  func.func @spn_vector(%arg0: memref<?x6xf32>, %arg1: memref<1x?xf32>) {
    %c0 = arith.constant 0 : index
    %0 = memref.dim %arg0, %c0 : memref<?x6xf32>
    %1 = memref.alloc(%0) : memref<1x?xf32>
    call @vec_task_0(%arg0, %1) : (memref<?x6xf32>, memref<1x?xf32>) -> ()
    %2 = memref.tensor_load %1 : memref<1x?xf32>
    %3 = memref.buffer_cast %2 : memref<1x?xf32>
    "lo_spn.copy"(%3, %arg1) : (memref<1x?xf32>, memref<1x?xf32>) -> ()
    "lo_spn.return"() : () -> ()
  }
}


// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

// CHECK-LABEL:   memref.global "private" arith.constant @histogram_1 : memref<2xf32> = dense<[-0.79850769, -0.597836971]>
// CHECK:         memref.global "private" arith.constant @histogram_0 : memref<2xf32> = dense<[-1.38629436, -0.287682086]>
// CHECK:         memref.global "private" arith.constant @categorical_1 : memref<3xf32> = dense<[-1.38629436, -0.470003635, -2.07944155]>
// CHECK:         memref.global "private" arith.constant @categorical_0 : memref<3xf32> = dense<[-1.04982209, -0.597836971, -2.30258512]>
// CHECK:         memref.global "private" arith.constant @histogram_vec_1 : memref<2xf32> = dense<[-0.79850769, -0.597836971]>
// CHECK:         memref.global "private" arith.constant @histogram_vec_0 : memref<2xf32> = dense<[-1.38629436, -0.287682086]>
// CHECK:         memref.global "private" arith.constant @categorical_vec_1 : memref<3xf32> = dense<[-1.38629436, -0.470003635, -2.07944155]>
// CHECK:         memref.global "private" arith.constant @categorical_vec_0 : memref<3xf32> = dense<[-1.04982209, -0.597836971, -2.30258512]>

// CHECK-LABEL:   func.func @vec_task_0(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<?x6xf32>,
// CHECK-SAME:                     %[[VAL_1:.*]]: memref<1x?xf32>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = memref.dim %[[VAL_0]], %[[VAL_2]] : memref<?x6xf32>
// CHECK:           %[[VAL_4:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_5:.*]] = remi_unsigned %[[VAL_3]], %[[VAL_4]] : index
// CHECK:           %[[VAL_6:.*]] = subi %[[VAL_3]], %[[VAL_5]] : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 4 : index
// CHECK:           scf.for %[[VAL_9:.*]] = %[[VAL_7]] to %[[VAL_6]] step %[[VAL_8]] {
// CHECK:             %[[VAL_10:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_11:.*]] = vector.broadcast %[[VAL_10]] : i64 to vector<8xi64>
// CHECK:             %[[VAL_12:.*]] = arith.constant dense<[0, 6, 12, 18, 24, 30, 36, 42]> : vector<8xi64>
// CHECK:             %[[VAL_13:.*]] = arith.constant dense<6> : vector<8xi64>
// CHECK:             %[[VAL_14:.*]] = muli %[[VAL_11]], %[[VAL_13]] : vector<8xi64>
// CHECK:             %[[VAL_15:.*]] = addi %[[VAL_14]], %[[VAL_12]] : vector<8xi64>
// CHECK:             %[[VAL_16:.*]] = arith.constant dense<0.000000e+00> : vector<8xf32>
// CHECK:             %[[VAL_17:.*]] = arith.constant dense<true> : vector<8xi1>
// CHECK:             %[[VAL_18:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_19:.*]] = memref.dim %[[VAL_0]], %[[VAL_18]] : memref<?x6xf32>
// CHECK:             %[[VAL_20:.*]] = arith.constant 6 : index
// CHECK:             %[[VAL_21:.*]] = muli %[[VAL_19]], %[[VAL_20]] : index
// CHECK:             %[[VAL_22:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_21]]], strides: [1] : memref<?x6xf32> to memref<?xf32>
// CHECK:             %[[VAL_23:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_24:.*]] = vector.gather %[[VAL_22]]{{\[}}%[[VAL_23]]] {{\[}}%[[VAL_15]]], %[[VAL_17]], %[[VAL_16]] : memref<?xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK:             %[[VAL_25:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_26:.*]] = vector.broadcast %[[VAL_25]] : i64 to vector<8xi64>
// CHECK:             %[[VAL_27:.*]] = arith.constant dense<[1, 7, 13, 19, 25, 31, 37, 43]> : vector<8xi64>
// CHECK:             %[[VAL_28:.*]] = arith.constant dense<6> : vector<8xi64>
// CHECK:             %[[VAL_29:.*]] = muli %[[VAL_26]], %[[VAL_28]] : vector<8xi64>
// CHECK:             %[[VAL_30:.*]] = addi %[[VAL_29]], %[[VAL_27]] : vector<8xi64>
// CHECK:             %[[VAL_31:.*]] = arith.constant dense<0.000000e+00> : vector<8xf32>
// CHECK:             %[[VAL_32:.*]] = arith.constant dense<true> : vector<8xi1>
// CHECK:             %[[VAL_33:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_34:.*]] = memref.dim %[[VAL_0]], %[[VAL_33]] : memref<?x6xf32>
// CHECK:             %[[VAL_35:.*]] = arith.constant 6 : index
// CHECK:             %[[VAL_36:.*]] = muli %[[VAL_34]], %[[VAL_35]] : index
// CHECK:             %[[VAL_37:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_36]]], strides: [1] : memref<?x6xf32> to memref<?xf32>
// CHECK:             %[[VAL_38:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_39:.*]] = vector.gather %[[VAL_37]]{{\[}}%[[VAL_38]]] {{\[}}%[[VAL_30]]], %[[VAL_32]], %[[VAL_31]] : memref<?xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK:             %[[VAL_40:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_41:.*]] = vector.broadcast %[[VAL_40]] : i64 to vector<8xi64>
// CHECK:             %[[VAL_42:.*]] = arith.constant dense<[2, 8, 14, 20, 26, 32, 38, 44]> : vector<8xi64>
// CHECK:             %[[VAL_43:.*]] = arith.constant dense<6> : vector<8xi64>
// CHECK:             %[[VAL_44:.*]] = muli %[[VAL_41]], %[[VAL_43]] : vector<8xi64>
// CHECK:             %[[VAL_45:.*]] = addi %[[VAL_44]], %[[VAL_42]] : vector<8xi64>
// CHECK:             %[[VAL_46:.*]] = arith.constant dense<0.000000e+00> : vector<8xf32>
// CHECK:             %[[VAL_47:.*]] = arith.constant dense<true> : vector<8xi1>
// CHECK:             %[[VAL_48:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_49:.*]] = memref.dim %[[VAL_0]], %[[VAL_48]] : memref<?x6xf32>
// CHECK:             %[[VAL_50:.*]] = arith.constant 6 : index
// CHECK:             %[[VAL_51:.*]] = muli %[[VAL_49]], %[[VAL_50]] : index
// CHECK:             %[[VAL_52:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_51]]], strides: [1] : memref<?x6xf32> to memref<?xf32>
// CHECK:             %[[VAL_53:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_54:.*]] = vector.gather %[[VAL_52]]{{\[}}%[[VAL_53]]] {{\[}}%[[VAL_45]]], %[[VAL_47]], %[[VAL_46]] : memref<?xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK:             %[[VAL_55:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_56:.*]] = vector.broadcast %[[VAL_55]] : i64 to vector<8xi64>
// CHECK:             %[[VAL_57:.*]] = arith.constant dense<[3, 9, 15, 21, 27, 33, 39, 45]> : vector<8xi64>
// CHECK:             %[[VAL_58:.*]] = arith.constant dense<6> : vector<8xi64>
// CHECK:             %[[VAL_59:.*]] = muli %[[VAL_56]], %[[VAL_58]] : vector<8xi64>
// CHECK:             %[[VAL_60:.*]] = addi %[[VAL_59]], %[[VAL_57]] : vector<8xi64>
// CHECK:             %[[VAL_61:.*]] = arith.constant dense<0.000000e+00> : vector<8xf32>
// CHECK:             %[[VAL_62:.*]] = arith.constant dense<true> : vector<8xi1>
// CHECK:             %[[VAL_63:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_64:.*]] = memref.dim %[[VAL_0]], %[[VAL_63]] : memref<?x6xf32>
// CHECK:             %[[VAL_65:.*]] = arith.constant 6 : index
// CHECK:             %[[VAL_66:.*]] = muli %[[VAL_64]], %[[VAL_65]] : index
// CHECK:             %[[VAL_67:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_66]]], strides: [1] : memref<?x6xf32> to memref<?xf32>
// CHECK:             %[[VAL_68:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_69:.*]] = vector.gather %[[VAL_67]]{{\[}}%[[VAL_68]]] {{\[}}%[[VAL_60]]], %[[VAL_62]], %[[VAL_61]] : memref<?xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK:             %[[VAL_70:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_71:.*]] = vector.broadcast %[[VAL_70]] : i64 to vector<8xi64>
// CHECK:             %[[VAL_72:.*]] = arith.constant dense<[4, 10, 16, 22, 28, 34, 40, 46]> : vector<8xi64>
// CHECK:             %[[VAL_73:.*]] = arith.constant dense<6> : vector<8xi64>
// CHECK:             %[[VAL_74:.*]] = muli %[[VAL_71]], %[[VAL_73]] : vector<8xi64>
// CHECK:             %[[VAL_75:.*]] = addi %[[VAL_74]], %[[VAL_72]] : vector<8xi64>
// CHECK:             %[[VAL_76:.*]] = arith.constant dense<0.000000e+00> : vector<8xf32>
// CHECK:             %[[VAL_77:.*]] = arith.constant dense<true> : vector<8xi1>
// CHECK:             %[[VAL_78:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_79:.*]] = memref.dim %[[VAL_0]], %[[VAL_78]] : memref<?x6xf32>
// CHECK:             %[[VAL_80:.*]] = arith.constant 6 : index
// CHECK:             %[[VAL_81:.*]] = muli %[[VAL_79]], %[[VAL_80]] : index
// CHECK:             %[[VAL_82:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_81]]], strides: [1] : memref<?x6xf32> to memref<?xf32>
// CHECK:             %[[VAL_83:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_84:.*]] = vector.gather %[[VAL_82]]{{\[}}%[[VAL_83]]] {{\[}}%[[VAL_75]]], %[[VAL_77]], %[[VAL_76]] : memref<?xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK:             %[[VAL_85:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_86:.*]] = vector.broadcast %[[VAL_85]] : i64 to vector<8xi64>
// CHECK:             %[[VAL_87:.*]] = arith.constant dense<[5, 11, 17, 23, 29, 35, 41, 47]> : vector<8xi64>
// CHECK:             %[[VAL_88:.*]] = arith.constant dense<6> : vector<8xi64>
// CHECK:             %[[VAL_89:.*]] = muli %[[VAL_86]], %[[VAL_88]] : vector<8xi64>
// CHECK:             %[[VAL_90:.*]] = addi %[[VAL_89]], %[[VAL_87]] : vector<8xi64>
// CHECK:             %[[VAL_91:.*]] = arith.constant dense<0.000000e+00> : vector<8xf32>
// CHECK:             %[[VAL_92:.*]] = arith.constant dense<true> : vector<8xi1>
// CHECK:             %[[VAL_93:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_94:.*]] = memref.dim %[[VAL_0]], %[[VAL_93]] : memref<?x6xf32>
// CHECK:             %[[VAL_95:.*]] = arith.constant 6 : index
// CHECK:             %[[VAL_96:.*]] = muli %[[VAL_94]], %[[VAL_95]] : index
// CHECK:             %[[VAL_97:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_96]]], strides: [1] : memref<?x6xf32> to memref<?xf32>
// CHECK:             %[[VAL_98:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_99:.*]] = vector.gather %[[VAL_97]]{{\[}}%[[VAL_98]]] {{\[}}%[[VAL_90]]], %[[VAL_92]], %[[VAL_91]] : memref<?xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK:             %[[VAL_100:.*]] = memref.get_global @categorical_vec_0 : memref<3xf32>
// CHECK:             %[[VAL_101:.*]] = fptoui %[[VAL_24]] : vector<8xf32> to vector<8xi64>
// CHECK:             %[[VAL_102:.*]] = arith.constant dense<0.000000e+00> : vector<8xf32>
// CHECK:             %[[VAL_103:.*]] = arith.constant dense<true> : vector<8xi1>
// CHECK:             %[[VAL_104:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_105:.*]] = vector.gather %[[VAL_100]]{{\[}}%[[VAL_104]]] {{\[}}%[[VAL_101]]], %[[VAL_103]], %[[VAL_102]] : memref<3xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK:             %[[VAL_106:.*]] = memref.get_global @categorical_vec_1 : memref<3xf32>
// CHECK:             %[[VAL_107:.*]] = fptoui %[[VAL_39]] : vector<8xf32> to vector<8xi64>
// CHECK:             %[[VAL_108:.*]] = arith.constant dense<0.000000e+00> : vector<8xf32>
// CHECK:             %[[VAL_109:.*]] = arith.constant dense<true> : vector<8xi1>
// CHECK:             %[[VAL_110:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_111:.*]] = vector.gather %[[VAL_106]]{{\[}}%[[VAL_110]]] {{\[}}%[[VAL_107]]], %[[VAL_109]], %[[VAL_108]] : memref<3xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK:             %[[VAL_112:.*]] = memref.get_global @histogram_vec_0 : memref<2xf32>
// CHECK:             %[[VAL_113:.*]] = fptoui %[[VAL_54]] : vector<8xf32> to vector<8xi64>
// CHECK:             %[[VAL_114:.*]] = arith.constant dense<0.000000e+00> : vector<8xf32>
// CHECK:             %[[VAL_115:.*]] = arith.constant dense<true> : vector<8xi1>
// CHECK:             %[[VAL_116:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_117:.*]] = vector.gather %[[VAL_112]]{{\[}}%[[VAL_116]]] {{\[}}%[[VAL_113]]], %[[VAL_115]], %[[VAL_114]] : memref<2xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK:             %[[VAL_118:.*]] = memref.get_global @histogram_vec_1 : memref<2xf32>
// CHECK:             %[[VAL_119:.*]] = fptoui %[[VAL_69]] : vector<8xf32> to vector<8xi64>
// CHECK:             %[[VAL_120:.*]] = arith.constant dense<0.000000e+00> : vector<8xf32>
// CHECK:             %[[VAL_121:.*]] = arith.constant dense<true> : vector<8xi1>
// CHECK:             %[[VAL_122:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_123:.*]] = vector.gather %[[VAL_118]]{{\[}}%[[VAL_122]]] {{\[}}%[[VAL_119]]], %[[VAL_121]], %[[VAL_120]] : memref<2xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK:             %[[VAL_124:.*]] = arith.constant dense<-5.000000e-01> : vector<8xf32>
// CHECK:             %[[VAL_125:.*]] = arith.constant dense<-0.918938517> : vector<8xf32>
// CHECK:             %[[VAL_126:.*]] = arith.constant dense<5.000000e-01> : vector<8xf32>
// CHECK:             %[[VAL_127:.*]] = subf %[[VAL_84]], %[[VAL_126]] : vector<8xf32>
// CHECK:             %[[VAL_128:.*]] = mulf %[[VAL_127]], %[[VAL_127]] : vector<8xf32>
// CHECK:             %[[VAL_129:.*]] = mulf %[[VAL_128]], %[[VAL_124]] : vector<8xf32>
// CHECK:             %[[VAL_130:.*]] = addf %[[VAL_125]], %[[VAL_129]] : vector<8xf32>
// CHECK:             %[[VAL_131:.*]] = arith.constant dense<-5.000000e+01> : vector<8xf32>
// CHECK:             %[[VAL_132:.*]] = arith.constant dense<1.38364661> : vector<8xf32>
// CHECK:             %[[VAL_133:.*]] = arith.constant dense<2.500000e-01> : vector<8xf32>
// CHECK:             %[[VAL_134:.*]] = subf %[[VAL_99]], %[[VAL_133]] : vector<8xf32>
// CHECK:             %[[VAL_135:.*]] = mulf %[[VAL_134]], %[[VAL_134]] : vector<8xf32>
// CHECK:             %[[VAL_136:.*]] = mulf %[[VAL_135]], %[[VAL_131]] : vector<8xf32>
// CHECK:             %[[VAL_137:.*]] = addf %[[VAL_132]], %[[VAL_136]] : vector<8xf32>
// CHECK:             %[[VAL_138:.*]] = addf %[[VAL_105]], %[[VAL_111]] : vector<8xf32>
// CHECK:             %[[VAL_139:.*]] = addf %[[VAL_138]], %[[VAL_117]] : vector<8xf32>
// CHECK:             %[[VAL_140:.*]] = arith.constant dense<1.000000e-01> : vector<8xf32>
// CHECK:             %[[VAL_141:.*]] = addf %[[VAL_139]], %[[VAL_140]] : vector<8xf32>
// CHECK:             %[[VAL_142:.*]] = addf %[[VAL_123]], %[[VAL_130]] : vector<8xf32>
// CHECK:             %[[VAL_143:.*]] = addf %[[VAL_142]], %[[VAL_137]] : vector<8xf32>
// CHECK:             %[[VAL_144:.*]] = arith.constant dense<1.000000e-01> : vector<8xf32>
// CHECK:             %[[VAL_145:.*]] = addf %[[VAL_143]], %[[VAL_144]] : vector<8xf32>
// CHECK:             %[[VAL_146:.*]] = cmpf ogt, %[[VAL_141]], %[[VAL_145]] : vector<8xf32>
// CHECK:             %[[VAL_147:.*]] = select %[[VAL_146]], %[[VAL_141]], %[[VAL_145]] : vector<8xi1>, vector<8xf32>
// CHECK:             %[[VAL_148:.*]] = select %[[VAL_146]], %[[VAL_145]], %[[VAL_141]] : vector<8xi1>, vector<8xf32>
// CHECK:             %[[VAL_149:.*]] = subf %[[VAL_148]], %[[VAL_147]] : vector<8xf32>
// CHECK:             %[[VAL_150:.*]] = math.exp %[[VAL_149]] : vector<8xf32>
// CHECK:             %[[VAL_151:.*]] = math.log1p %[[VAL_150]] : vector<8xf32>
// CHECK:             %[[VAL_152:.*]] = addf %[[VAL_147]], %[[VAL_151]] : vector<8xf32>
// CHECK:             %[[VAL_153:.*]] = arith.constant 0 : index
// CHECK:             vector.transfer_write %[[VAL_152]], %[[VAL_1]]{{\[}}%[[VAL_153]], %[[VAL_9]]] : vector<8xf32>, memref<1x?xf32>
// CHECK:           }
// CHECK:           %[[VAL_154:.*]] = arith.constant 1 : index
// CHECK:           scf.for %[[VAL_155:.*]] = %[[VAL_6]] to %[[VAL_3]] step %[[VAL_154]] {
// CHECK:             %[[VAL_156:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_157:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_155]], %[[VAL_156]]] : memref<?x6xf32>
// CHECK:             %[[VAL_158:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_159:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_155]], %[[VAL_158]]] : memref<?x6xf32>
// CHECK:             %[[VAL_160:.*]] = arith.constant 2 : index
// CHECK:             %[[VAL_161:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_155]], %[[VAL_160]]] : memref<?x6xf32>
// CHECK:             %[[VAL_162:.*]] = arith.constant 3 : index
// CHECK:             %[[VAL_163:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_155]], %[[VAL_162]]] : memref<?x6xf32>
// CHECK:             %[[VAL_164:.*]] = arith.constant 4 : index
// CHECK:             %[[VAL_165:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_155]], %[[VAL_164]]] : memref<?x6xf32>
// CHECK:             %[[VAL_166:.*]] = arith.constant 5 : index
// CHECK:             %[[VAL_167:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_155]], %[[VAL_166]]] : memref<?x6xf32>
// CHECK:             %[[VAL_168:.*]] = memref.get_global @categorical_0 : memref<3xf32>
// CHECK:             %[[VAL_169:.*]] = fptoui %[[VAL_157]] : f32 to i64
// CHECK:             %[[VAL_170:.*]] = index_cast %[[VAL_169]] : i64 to index
// CHECK:             %[[VAL_171:.*]] = memref.load %[[VAL_168]]{{\[}}%[[VAL_170]]] : memref<3xf32>
// CHECK:             %[[VAL_172:.*]] = memref.get_global @categorical_1 : memref<3xf32>
// CHECK:             %[[VAL_173:.*]] = fptoui %[[VAL_159]] : f32 to i64
// CHECK:             %[[VAL_174:.*]] = index_cast %[[VAL_173]] : i64 to index
// CHECK:             %[[VAL_175:.*]] = memref.load %[[VAL_172]]{{\[}}%[[VAL_174]]] : memref<3xf32>
// CHECK:             %[[VAL_176:.*]] = memref.get_global @histogram_0 : memref<2xf32>
// CHECK:             %[[VAL_177:.*]] = fptoui %[[VAL_161]] : f32 to i64
// CHECK:             %[[VAL_178:.*]] = index_cast %[[VAL_177]] : i64 to index
// CHECK:             %[[VAL_179:.*]] = memref.load %[[VAL_176]]{{\[}}%[[VAL_178]]] : memref<2xf32>
// CHECK:             %[[VAL_180:.*]] = memref.get_global @histogram_1 : memref<2xf32>
// CHECK:             %[[VAL_181:.*]] = fptoui %[[VAL_163]] : f32 to i64
// CHECK:             %[[VAL_182:.*]] = index_cast %[[VAL_181]] : i64 to index
// CHECK:             %[[VAL_183:.*]] = memref.load %[[VAL_180]]{{\[}}%[[VAL_182]]] : memref<2xf32>
// CHECK:             %[[VAL_184:.*]] = arith.constant -5.000000e-01 : f32
// CHECK:             %[[VAL_185:.*]] = arith.constant -0.918938517 : f32
// CHECK:             %[[VAL_186:.*]] = arith.constant 5.000000e-01 : f32
// CHECK:             %[[VAL_187:.*]] = subf %[[VAL_165]], %[[VAL_186]] : f32
// CHECK:             %[[VAL_188:.*]] = mulf %[[VAL_187]], %[[VAL_187]] : f32
// CHECK:             %[[VAL_189:.*]] = mulf %[[VAL_188]], %[[VAL_184]] : f32
// CHECK:             %[[VAL_190:.*]] = addf %[[VAL_185]], %[[VAL_189]] : f32
// CHECK:             %[[VAL_191:.*]] = arith.constant -5.000000e+01 : f32
// CHECK:             %[[VAL_192:.*]] = arith.constant 1.38364661 : f32
// CHECK:             %[[VAL_193:.*]] = arith.constant 2.500000e-01 : f32
// CHECK:             %[[VAL_194:.*]] = subf %[[VAL_167]], %[[VAL_193]] : f32
// CHECK:             %[[VAL_195:.*]] = mulf %[[VAL_194]], %[[VAL_194]] : f32
// CHECK:             %[[VAL_196:.*]] = mulf %[[VAL_195]], %[[VAL_191]] : f32
// CHECK:             %[[VAL_197:.*]] = addf %[[VAL_192]], %[[VAL_196]] : f32
// CHECK:             %[[VAL_198:.*]] = addf %[[VAL_171]], %[[VAL_175]] : f32
// CHECK:             %[[VAL_199:.*]] = addf %[[VAL_198]], %[[VAL_179]] : f32
// CHECK:             %[[VAL_200:.*]] = arith.constant 1.000000e-01 : f32
// CHECK:             %[[VAL_201:.*]] = addf %[[VAL_199]], %[[VAL_200]] : f32
// CHECK:             %[[VAL_202:.*]] = addf %[[VAL_183]], %[[VAL_190]] : f32
// CHECK:             %[[VAL_203:.*]] = addf %[[VAL_202]], %[[VAL_197]] : f32
// CHECK:             %[[VAL_204:.*]] = arith.constant 1.000000e-01 : f32
// CHECK:             %[[VAL_205:.*]] = addf %[[VAL_203]], %[[VAL_204]] : f32
// CHECK:             %[[VAL_206:.*]] = cmpf ogt, %[[VAL_201]], %[[VAL_205]] : f32
// CHECK:             %[[VAL_207:.*]] = select %[[VAL_206]], %[[VAL_201]], %[[VAL_205]] : f32
// CHECK:             %[[VAL_208:.*]] = select %[[VAL_206]], %[[VAL_205]], %[[VAL_201]] : f32
// CHECK:             %[[VAL_209:.*]] = subf %[[VAL_208]], %[[VAL_207]] : f32
// CHECK:             %[[VAL_210:.*]] = math.exp %[[VAL_209]] : f32
// CHECK:             %[[VAL_211:.*]] = math.log1p %[[VAL_210]] : f32
// CHECK:             %[[VAL_212:.*]] = addf %[[VAL_207]], %[[VAL_211]] : f32
// CHECK:             %[[VAL_213:.*]] = arith.constant 0 : index
// CHECK:             memref.store %[[VAL_212]], %[[VAL_1]]{{\[}}%[[VAL_213]], %[[VAL_155]]] : memref<1x?xf32>
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @spn_vector(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<?x6xf32>,
// CHECK-SAME:                     %[[VAL_1:.*]]: memref<1x?xf32>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = memref.dim %[[VAL_0]], %[[VAL_2]] : memref<?x6xf32>
// CHECK:           %[[VAL_4:.*]] = memref.alloc(%[[VAL_3]]) : memref<1x?xf32>
// CHECK:           call @vec_task_0(%[[VAL_0]], %[[VAL_4]]) : (memref<?x6xf32>, memref<1x?xf32>) -> ()
// CHECK:           %[[VAL_5:.*]] = memref.tensor_load %[[VAL_4]] : memref<1x?xf32>
// CHECK:           %[[VAL_6:.*]] = memref.buffer_cast %[[VAL_5]] : memref<1x?xf32>
// CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_8:.*]] = memref.dim %[[VAL_6]], %[[VAL_7]] : memref<1x?xf32>
// CHECK:           %[[VAL_9:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_10:.*]] = arith.constant 1 : index
// CHECK:           scf.for %[[VAL_11:.*]] = %[[VAL_9]] to %[[VAL_8]] step %[[VAL_10]] {
// CHECK:             %[[VAL_12:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_13:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_12]], %[[VAL_11]]] : memref<1x?xf32>
// CHECK:             memref.store %[[VAL_13]], %[[VAL_1]]{{\[}}%[[VAL_12]], %[[VAL_11]]] : memref<1x?xf32>
// CHECK:           }
// CHECK:           return
// CHECK:         }
