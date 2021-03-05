// RUN: %optcall --convert-lospn-nodes-to-cpu %s | FileCheck %s

module  {
  global_memref "private" constant @histogram_vec_1 : memref<2xf32> = dense<[-0.79850769, -0.597836971]>
  global_memref "private" constant @histogram_vec_0 : memref<2xf32> = dense<[-1.38629436, -0.287682086]>
  global_memref "private" constant @categorical_vec_1 : memref<3xf32> = dense<[-1.38629436, -0.470003635, -2.07944155]>
  global_memref "private" constant @categorical_vec_0 : memref<3xf32> = dense<[-1.04982209, -0.597836971, -2.30258512]>
  func @vec_task_0(%arg0: memref<?x6xf32>, %arg1: memref<?xf32>) {
    %c0 = constant 0 : index
    %0 = dim %arg0, %c0 : memref<?x6xf32>
    %c4 = constant 4 : index
    %1 = remi_unsigned %0, %c4 : index
    %2 = subi %0, %1 : index
    %c0_0 = constant 0 : index
    %c4_1 = constant 4 : index
    scf.for %arg2 = %c0_0 to %2 step %c4_1 {
      %3 = index_cast %arg2 : index to i64
      %4 = vector.broadcast %3 : i64 to vector<8xi64>
      %cst = constant dense<[0, 6, 12, 18, 24, 30, 36, 42]> : vector<8xi64>
      %cst_2 = constant dense<6> : vector<8xi64>
      %5 = muli %4, %cst_2 : vector<8xi64>
      %6 = addi %5, %cst : vector<8xi64>
      %cst_3 = constant dense<0.000000e+00> : vector<8xf32>
      %cst_4 = constant dense<true> : vector<8xi1>
      %c0_5 = constant 0 : index
      %7 = dim %arg0, %c0_5 : memref<?x6xf32>
      %c6 = constant 6 : index
      %8 = muli %7, %c6 : index
      %9 = memref_reinterpret_cast %arg0 to offset: [0], sizes: [%8], strides: [1] : memref<?x6xf32> to memref<?xf32>
      %10 = vector.gather %9[%6], %cst_4, %cst_3 : memref<?xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
      %11 = index_cast %arg2 : index to i64
      %12 = vector.broadcast %11 : i64 to vector<8xi64>
      %cst_6 = constant dense<[1, 7, 13, 19, 25, 31, 37, 43]> : vector<8xi64>
      %cst_7 = constant dense<6> : vector<8xi64>
      %13 = muli %12, %cst_7 : vector<8xi64>
      %14 = addi %13, %cst_6 : vector<8xi64>
      %cst_8 = constant dense<0.000000e+00> : vector<8xf32>
      %cst_9 = constant dense<true> : vector<8xi1>
      %c0_10 = constant 0 : index
      %15 = dim %arg0, %c0_10 : memref<?x6xf32>
      %c6_11 = constant 6 : index
      %16 = muli %15, %c6_11 : index
      %17 = memref_reinterpret_cast %arg0 to offset: [0], sizes: [%16], strides: [1] : memref<?x6xf32> to memref<?xf32>
      %18 = vector.gather %17[%14], %cst_9, %cst_8 : memref<?xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
      %19 = index_cast %arg2 : index to i64
      %20 = vector.broadcast %19 : i64 to vector<8xi64>
      %cst_12 = constant dense<[2, 8, 14, 20, 26, 32, 38, 44]> : vector<8xi64>
      %cst_13 = constant dense<6> : vector<8xi64>
      %21 = muli %20, %cst_13 : vector<8xi64>
      %22 = addi %21, %cst_12 : vector<8xi64>
      %cst_14 = constant dense<0.000000e+00> : vector<8xf32>
      %cst_15 = constant dense<true> : vector<8xi1>
      %c0_16 = constant 0 : index
      %23 = dim %arg0, %c0_16 : memref<?x6xf32>
      %c6_17 = constant 6 : index
      %24 = muli %23, %c6_17 : index
      %25 = memref_reinterpret_cast %arg0 to offset: [0], sizes: [%24], strides: [1] : memref<?x6xf32> to memref<?xf32>
      %26 = vector.gather %25[%22], %cst_15, %cst_14 : memref<?xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
      %27 = index_cast %arg2 : index to i64
      %28 = vector.broadcast %27 : i64 to vector<8xi64>
      %cst_18 = constant dense<[3, 9, 15, 21, 27, 33, 39, 45]> : vector<8xi64>
      %cst_19 = constant dense<6> : vector<8xi64>
      %29 = muli %28, %cst_19 : vector<8xi64>
      %30 = addi %29, %cst_18 : vector<8xi64>
      %cst_20 = constant dense<0.000000e+00> : vector<8xf32>
      %cst_21 = constant dense<true> : vector<8xi1>
      %c0_22 = constant 0 : index
      %31 = dim %arg0, %c0_22 : memref<?x6xf32>
      %c6_23 = constant 6 : index
      %32 = muli %31, %c6_23 : index
      %33 = memref_reinterpret_cast %arg0 to offset: [0], sizes: [%32], strides: [1] : memref<?x6xf32> to memref<?xf32>
      %34 = vector.gather %33[%30], %cst_21, %cst_20 : memref<?xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
      %35 = index_cast %arg2 : index to i64
      %36 = vector.broadcast %35 : i64 to vector<8xi64>
      %cst_24 = constant dense<[4, 10, 16, 22, 28, 34, 40, 46]> : vector<8xi64>
      %cst_25 = constant dense<6> : vector<8xi64>
      %37 = muli %36, %cst_25 : vector<8xi64>
      %38 = addi %37, %cst_24 : vector<8xi64>
      %cst_26 = constant dense<0.000000e+00> : vector<8xf32>
      %cst_27 = constant dense<true> : vector<8xi1>
      %c0_28 = constant 0 : index
      %39 = dim %arg0, %c0_28 : memref<?x6xf32>
      %c6_29 = constant 6 : index
      %40 = muli %39, %c6_29 : index
      %41 = memref_reinterpret_cast %arg0 to offset: [0], sizes: [%40], strides: [1] : memref<?x6xf32> to memref<?xf32>
      %42 = vector.gather %41[%38], %cst_27, %cst_26 : memref<?xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
      %43 = index_cast %arg2 : index to i64
      %44 = vector.broadcast %43 : i64 to vector<8xi64>
      %cst_30 = constant dense<[5, 11, 17, 23, 29, 35, 41, 47]> : vector<8xi64>
      %cst_31 = constant dense<6> : vector<8xi64>
      %45 = muli %44, %cst_31 : vector<8xi64>
      %46 = addi %45, %cst_30 : vector<8xi64>
      %cst_32 = constant dense<0.000000e+00> : vector<8xf32>
      %cst_33 = constant dense<true> : vector<8xi1>
      %c0_34 = constant 0 : index
      %47 = dim %arg0, %c0_34 : memref<?x6xf32>
      %c6_35 = constant 6 : index
      %48 = muli %47, %c6_35 : index
      %49 = memref_reinterpret_cast %arg0 to offset: [0], sizes: [%48], strides: [1] : memref<?x6xf32> to memref<?xf32>
      %50 = vector.gather %49[%46], %cst_33, %cst_32 : memref<?xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
      %51 = get_global_memref @categorical_vec_0 : memref<3xf32>
      %52 = fptoui %10 : vector<8xf32> to vector<8xi64>
      %cst_36 = constant dense<0.000000e+00> : vector<8xf32>
      %cst_37 = constant dense<true> : vector<8xi1>
      %53 = vector.gather %51[%52], %cst_37, %cst_36 : memref<3xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
      %54 = get_global_memref @categorical_vec_1 : memref<3xf32>
      %55 = fptoui %18 : vector<8xf32> to vector<8xi64>
      %cst_38 = constant dense<0.000000e+00> : vector<8xf32>
      %cst_39 = constant dense<true> : vector<8xi1>
      %56 = vector.gather %54[%55], %cst_39, %cst_38 : memref<3xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
      %57 = get_global_memref @histogram_vec_0 : memref<2xf32>
      %58 = fptoui %26 : vector<8xf32> to vector<8xi64>
      %cst_40 = constant dense<0.000000e+00> : vector<8xf32>
      %cst_41 = constant dense<true> : vector<8xi1>
      %59 = vector.gather %57[%58], %cst_41, %cst_40 : memref<2xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
      %60 = get_global_memref @histogram_vec_1 : memref<2xf32>
      %61 = fptoui %34 : vector<8xf32> to vector<8xi64>
      %cst_42 = constant dense<0.000000e+00> : vector<8xf32>
      %cst_43 = constant dense<true> : vector<8xi1>
      %62 = vector.gather %60[%61], %cst_43, %cst_42 : memref<2xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
      %cst_44 = constant dense<-5.000000e-01> : vector<8xf32>
      %cst_45 = constant dense<-0.918938517> : vector<8xf32>
      %cst_46 = constant dense<5.000000e-01> : vector<8xf32>
      %63 = subf %42, %cst_46 : vector<8xf32>
      %64 = mulf %63, %63 : vector<8xf32>
      %65 = mulf %64, %cst_44 : vector<8xf32>
      %66 = addf %cst_45, %65 : vector<8xf32>
      %cst_47 = constant dense<-5.000000e+01> : vector<8xf32>
      %cst_48 = constant dense<1.38364661> : vector<8xf32>
      %cst_49 = constant dense<2.500000e-01> : vector<8xf32>
      %67 = subf %50, %cst_49 : vector<8xf32>
      %68 = mulf %67, %67 : vector<8xf32>
      %69 = mulf %68, %cst_47 : vector<8xf32>
      %70 = addf %cst_48, %69 : vector<8xf32>
      %71 = addf %53, %56 : vector<8xf32>
      %72 = addf %71, %59 : vector<8xf32>
      %cst_50 = constant dense<1.000000e-01> : vector<8xf32>
      %73 = addf %72, %cst_50 : vector<8xf32>
      %74 = addf %62, %66 : vector<8xf32>
      %75 = addf %74, %70 : vector<8xf32>
      %cst_51 = constant dense<1.000000e-01> : vector<8xf32>
      %76 = addf %75, %cst_51 : vector<8xf32>
      %77 = cmpf ogt, %73, %76 : vector<8xf32>
      %78 = select %77, %73, %76 : vector<8xi1>, vector<8xf32>
      %79 = select %77, %76, %73 : vector<8xi1>, vector<8xf32>
      %80 = subf %79, %78 : vector<8xf32>
      %81 = math.exp %80 : vector<8xf32>
      %cst_52 = constant dense<1.000000e+00> : vector<8xf32>
      %82 = addf %cst_52, %81 : vector<8xf32>
      %83 = math.log %82 : vector<8xf32>
      %84 = addf %78, %83 : vector<8xf32>
      vector.transfer_write %84, %arg1[%arg2] : vector<8xf32>, memref<?xf32>
    }
    %c1 = constant 1 : index
    scf.for %arg2 = %2 to %0 step %c1 {
      %3 = "lo_spn.batch_read"(%arg0, %arg2) {sampleIndex = 0 : ui32} : (memref<?x6xf32>, index) -> f32
      %4 = "lo_spn.batch_read"(%arg0, %arg2) {sampleIndex = 1 : ui32} : (memref<?x6xf32>, index) -> f32
      %5 = "lo_spn.batch_read"(%arg0, %arg2) {sampleIndex = 2 : ui32} : (memref<?x6xf32>, index) -> f32
      %6 = "lo_spn.batch_read"(%arg0, %arg2) {sampleIndex = 3 : ui32} : (memref<?x6xf32>, index) -> f32
      %7 = "lo_spn.batch_read"(%arg0, %arg2) {sampleIndex = 4 : ui32} : (memref<?x6xf32>, index) -> f32
      %8 = "lo_spn.batch_read"(%arg0, %arg2) {sampleIndex = 5 : ui32} : (memref<?x6xf32>, index) -> f32
      %9 = "lo_spn.categorical"(%3) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false} : (f32) -> !lo_spn.log<f32>
      %10 = "lo_spn.categorical"(%4) {probabilities = [2.500000e-01, 6.250000e-01, 1.250000e-01], supportMarginal = false} : (f32) -> !lo_spn.log<f32>
      %11 = "lo_spn.histogram"(%5) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}], supportMarginal = false} : (f32) -> !lo_spn.log<f32>
      %12 = "lo_spn.histogram"(%6) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 4.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 5.500000e-01 : f64}], supportMarginal = false} : (f32) -> !lo_spn.log<f32>
      %13 = "lo_spn.gaussian"(%7) {mean = 5.000000e-01 : f64, stddev = 1.000000e+00 : f64, supportMarginal = false} : (f32) -> !lo_spn.log<f32>
      %14 = "lo_spn.gaussian"(%8) {mean = 2.500000e-01 : f64, stddev = 1.000000e-01 : f64, supportMarginal = false} : (f32) -> !lo_spn.log<f32>
      %15 = "lo_spn.mul"(%9, %10) : (!lo_spn.log<f32>, !lo_spn.log<f32>) -> !lo_spn.log<f32>
      %16 = "lo_spn.mul"(%15, %11) : (!lo_spn.log<f32>, !lo_spn.log<f32>) -> !lo_spn.log<f32>
      %17 = "lo_spn.constant"() {type = !lo_spn.log<f32>, value = 1.000000e-01 : f64} : () -> !lo_spn.log<f32>
      %18 = "lo_spn.mul"(%16, %17) : (!lo_spn.log<f32>, !lo_spn.log<f32>) -> !lo_spn.log<f32>
      %19 = "lo_spn.mul"(%12, %13) : (!lo_spn.log<f32>, !lo_spn.log<f32>) -> !lo_spn.log<f32>
      %20 = "lo_spn.mul"(%19, %14) : (!lo_spn.log<f32>, !lo_spn.log<f32>) -> !lo_spn.log<f32>
      %21 = "lo_spn.constant"() {type = !lo_spn.log<f32>, value = 1.000000e-01 : f64} : () -> !lo_spn.log<f32>
      %22 = "lo_spn.mul"(%20, %21) : (!lo_spn.log<f32>, !lo_spn.log<f32>) -> !lo_spn.log<f32>
      %23 = "lo_spn.add"(%18, %22) : (!lo_spn.log<f32>, !lo_spn.log<f32>) -> !lo_spn.log<f32>
      %24 = "lo_spn.strip_log"(%23) {target = f32} : (!lo_spn.log<f32>) -> f32
      "lo_spn.batch_write"(%24, %arg1, %arg2) : (f32, memref<?xf32>, index) -> ()
    }
    return
  }
  func @spn_vector(%arg0: memref<?x6xf32>, %arg1: memref<?xf32>) {
    %c0 = constant 0 : index
    %0 = dim %arg0, %c0 : memref<?x6xf32>
    %1 = alloc(%0) : memref<?xf32>
    call @vec_task_0(%arg0, %1) : (memref<?x6xf32>, memref<?xf32>) -> ()
    %2 = tensor_load %1 : memref<?xf32>
    %3 = tensor_to_memref %2 : memref<?xf32>
    "lo_spn.copy"(%3, %arg1) : (memref<?xf32>, memref<?xf32>) -> ()
    "lo_spn.return"() : () -> ()
  }
}

// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

// CHECK-LABEL:   global_memref "private" constant @histogram_1 : memref<2xf32> = dense<[-0.79850769, -0.597836971]>
// CHECK:         global_memref "private" constant @histogram_0 : memref<2xf32> = dense<[-1.38629436, -0.287682086]>
// CHECK:         global_memref "private" constant @categorical_1 : memref<3xf32> = dense<[-1.38629436, -0.470003635, -2.07944155]>
// CHECK:         global_memref "private" constant @categorical_0 : memref<3xf32> = dense<[-1.04982209, -0.597836971, -2.30258512]>
// CHECK:         global_memref "private" constant @histogram_vec_1 : memref<2xf32> = dense<[-0.79850769, -0.597836971]>
// CHECK:         global_memref "private" constant @histogram_vec_0 : memref<2xf32> = dense<[-1.38629436, -0.287682086]>
// CHECK:         global_memref "private" constant @categorical_vec_1 : memref<3xf32> = dense<[-1.38629436, -0.470003635, -2.07944155]>
// CHECK:         global_memref "private" constant @categorical_vec_0 : memref<3xf32> = dense<[-1.04982209, -0.597836971, -2.30258512]>

// CHECK-LABEL:   func @vec_task_0(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<?x6xf32>,
// CHECK-SAME:                     %[[VAL_1:.*]]: memref<?xf32>) {
// CHECK:           %[[VAL_2:.*]] = constant 0 : index
// CHECK:           %[[VAL_3:.*]] = dim %[[VAL_0]], %[[VAL_2]] : memref<?x6xf32>
// CHECK:           %[[VAL_4:.*]] = constant 4 : index
// CHECK:           %[[VAL_5:.*]] = remi_unsigned %[[VAL_3]], %[[VAL_4]] : index
// CHECK:           %[[VAL_6:.*]] = subi %[[VAL_3]], %[[VAL_5]] : index
// CHECK:           %[[VAL_7:.*]] = constant 0 : index
// CHECK:           %[[VAL_8:.*]] = constant 4 : index
// CHECK:           scf.for %[[VAL_9:.*]] = %[[VAL_7]] to %[[VAL_6]] step %[[VAL_8]] {
// CHECK:             %[[VAL_10:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_11:.*]] = vector.broadcast %[[VAL_10]] : i64 to vector<8xi64>
// CHECK:             %[[VAL_12:.*]] = constant dense<[0, 6, 12, 18, 24, 30, 36, 42]> : vector<8xi64>
// CHECK:             %[[VAL_13:.*]] = constant dense<6> : vector<8xi64>
// CHECK:             %[[VAL_14:.*]] = muli %[[VAL_11]], %[[VAL_13]] : vector<8xi64>
// CHECK:             %[[VAL_15:.*]] = addi %[[VAL_14]], %[[VAL_12]] : vector<8xi64>
// CHECK:             %[[VAL_16:.*]] = constant dense<0.000000e+00> : vector<8xf32>
// CHECK:             %[[VAL_17:.*]] = constant dense<true> : vector<8xi1>
// CHECK:             %[[VAL_18:.*]] = constant 0 : index
// CHECK:             %[[VAL_19:.*]] = dim %[[VAL_0]], %[[VAL_18]] : memref<?x6xf32>
// CHECK:             %[[VAL_20:.*]] = constant 6 : index
// CHECK:             %[[VAL_21:.*]] = muli %[[VAL_19]], %[[VAL_20]] : index
// CHECK:             %[[VAL_22:.*]] = memref_reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_21]]], strides: [1] : memref<?x6xf32> to memref<?xf32>
// CHECK:             %[[VAL_23:.*]] = vector.gather %[[VAL_22]]{{\[}}%[[VAL_15]]], %[[VAL_17]], %[[VAL_16]] : memref<?xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK:             %[[VAL_24:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_25:.*]] = vector.broadcast %[[VAL_24]] : i64 to vector<8xi64>
// CHECK:             %[[VAL_26:.*]] = constant dense<[1, 7, 13, 19, 25, 31, 37, 43]> : vector<8xi64>
// CHECK:             %[[VAL_27:.*]] = constant dense<6> : vector<8xi64>
// CHECK:             %[[VAL_28:.*]] = muli %[[VAL_25]], %[[VAL_27]] : vector<8xi64>
// CHECK:             %[[VAL_29:.*]] = addi %[[VAL_28]], %[[VAL_26]] : vector<8xi64>
// CHECK:             %[[VAL_30:.*]] = constant dense<0.000000e+00> : vector<8xf32>
// CHECK:             %[[VAL_31:.*]] = constant dense<true> : vector<8xi1>
// CHECK:             %[[VAL_32:.*]] = constant 0 : index
// CHECK:             %[[VAL_33:.*]] = dim %[[VAL_0]], %[[VAL_32]] : memref<?x6xf32>
// CHECK:             %[[VAL_34:.*]] = constant 6 : index
// CHECK:             %[[VAL_35:.*]] = muli %[[VAL_33]], %[[VAL_34]] : index
// CHECK:             %[[VAL_36:.*]] = memref_reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_35]]], strides: [1] : memref<?x6xf32> to memref<?xf32>
// CHECK:             %[[VAL_37:.*]] = vector.gather %[[VAL_36]]{{\[}}%[[VAL_29]]], %[[VAL_31]], %[[VAL_30]] : memref<?xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK:             %[[VAL_38:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_39:.*]] = vector.broadcast %[[VAL_38]] : i64 to vector<8xi64>
// CHECK:             %[[VAL_40:.*]] = constant dense<[2, 8, 14, 20, 26, 32, 38, 44]> : vector<8xi64>
// CHECK:             %[[VAL_41:.*]] = constant dense<6> : vector<8xi64>
// CHECK:             %[[VAL_42:.*]] = muli %[[VAL_39]], %[[VAL_41]] : vector<8xi64>
// CHECK:             %[[VAL_43:.*]] = addi %[[VAL_42]], %[[VAL_40]] : vector<8xi64>
// CHECK:             %[[VAL_44:.*]] = constant dense<0.000000e+00> : vector<8xf32>
// CHECK:             %[[VAL_45:.*]] = constant dense<true> : vector<8xi1>
// CHECK:             %[[VAL_46:.*]] = constant 0 : index
// CHECK:             %[[VAL_47:.*]] = dim %[[VAL_0]], %[[VAL_46]] : memref<?x6xf32>
// CHECK:             %[[VAL_48:.*]] = constant 6 : index
// CHECK:             %[[VAL_49:.*]] = muli %[[VAL_47]], %[[VAL_48]] : index
// CHECK:             %[[VAL_50:.*]] = memref_reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_49]]], strides: [1] : memref<?x6xf32> to memref<?xf32>
// CHECK:             %[[VAL_51:.*]] = vector.gather %[[VAL_50]]{{\[}}%[[VAL_43]]], %[[VAL_45]], %[[VAL_44]] : memref<?xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK:             %[[VAL_52:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_53:.*]] = vector.broadcast %[[VAL_52]] : i64 to vector<8xi64>
// CHECK:             %[[VAL_54:.*]] = constant dense<[3, 9, 15, 21, 27, 33, 39, 45]> : vector<8xi64>
// CHECK:             %[[VAL_55:.*]] = constant dense<6> : vector<8xi64>
// CHECK:             %[[VAL_56:.*]] = muli %[[VAL_53]], %[[VAL_55]] : vector<8xi64>
// CHECK:             %[[VAL_57:.*]] = addi %[[VAL_56]], %[[VAL_54]] : vector<8xi64>
// CHECK:             %[[VAL_58:.*]] = constant dense<0.000000e+00> : vector<8xf32>
// CHECK:             %[[VAL_59:.*]] = constant dense<true> : vector<8xi1>
// CHECK:             %[[VAL_60:.*]] = constant 0 : index
// CHECK:             %[[VAL_61:.*]] = dim %[[VAL_0]], %[[VAL_60]] : memref<?x6xf32>
// CHECK:             %[[VAL_62:.*]] = constant 6 : index
// CHECK:             %[[VAL_63:.*]] = muli %[[VAL_61]], %[[VAL_62]] : index
// CHECK:             %[[VAL_64:.*]] = memref_reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_63]]], strides: [1] : memref<?x6xf32> to memref<?xf32>
// CHECK:             %[[VAL_65:.*]] = vector.gather %[[VAL_64]]{{\[}}%[[VAL_57]]], %[[VAL_59]], %[[VAL_58]] : memref<?xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK:             %[[VAL_66:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_67:.*]] = vector.broadcast %[[VAL_66]] : i64 to vector<8xi64>
// CHECK:             %[[VAL_68:.*]] = constant dense<[4, 10, 16, 22, 28, 34, 40, 46]> : vector<8xi64>
// CHECK:             %[[VAL_69:.*]] = constant dense<6> : vector<8xi64>
// CHECK:             %[[VAL_70:.*]] = muli %[[VAL_67]], %[[VAL_69]] : vector<8xi64>
// CHECK:             %[[VAL_71:.*]] = addi %[[VAL_70]], %[[VAL_68]] : vector<8xi64>
// CHECK:             %[[VAL_72:.*]] = constant dense<0.000000e+00> : vector<8xf32>
// CHECK:             %[[VAL_73:.*]] = constant dense<true> : vector<8xi1>
// CHECK:             %[[VAL_74:.*]] = constant 0 : index
// CHECK:             %[[VAL_75:.*]] = dim %[[VAL_0]], %[[VAL_74]] : memref<?x6xf32>
// CHECK:             %[[VAL_76:.*]] = constant 6 : index
// CHECK:             %[[VAL_77:.*]] = muli %[[VAL_75]], %[[VAL_76]] : index
// CHECK:             %[[VAL_78:.*]] = memref_reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_77]]], strides: [1] : memref<?x6xf32> to memref<?xf32>
// CHECK:             %[[VAL_79:.*]] = vector.gather %[[VAL_78]]{{\[}}%[[VAL_71]]], %[[VAL_73]], %[[VAL_72]] : memref<?xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK:             %[[VAL_80:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_81:.*]] = vector.broadcast %[[VAL_80]] : i64 to vector<8xi64>
// CHECK:             %[[VAL_82:.*]] = constant dense<[5, 11, 17, 23, 29, 35, 41, 47]> : vector<8xi64>
// CHECK:             %[[VAL_83:.*]] = constant dense<6> : vector<8xi64>
// CHECK:             %[[VAL_84:.*]] = muli %[[VAL_81]], %[[VAL_83]] : vector<8xi64>
// CHECK:             %[[VAL_85:.*]] = addi %[[VAL_84]], %[[VAL_82]] : vector<8xi64>
// CHECK:             %[[VAL_86:.*]] = constant dense<0.000000e+00> : vector<8xf32>
// CHECK:             %[[VAL_87:.*]] = constant dense<true> : vector<8xi1>
// CHECK:             %[[VAL_88:.*]] = constant 0 : index
// CHECK:             %[[VAL_89:.*]] = dim %[[VAL_0]], %[[VAL_88]] : memref<?x6xf32>
// CHECK:             %[[VAL_90:.*]] = constant 6 : index
// CHECK:             %[[VAL_91:.*]] = muli %[[VAL_89]], %[[VAL_90]] : index
// CHECK:             %[[VAL_92:.*]] = memref_reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_91]]], strides: [1] : memref<?x6xf32> to memref<?xf32>
// CHECK:             %[[VAL_93:.*]] = vector.gather %[[VAL_92]]{{\[}}%[[VAL_85]]], %[[VAL_87]], %[[VAL_86]] : memref<?xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK:             %[[VAL_94:.*]] = get_global_memref @categorical_vec_0 : memref<3xf32>
// CHECK:             %[[VAL_95:.*]] = fptoui %[[VAL_23]] : vector<8xf32> to vector<8xi64>
// CHECK:             %[[VAL_96:.*]] = constant dense<0.000000e+00> : vector<8xf32>
// CHECK:             %[[VAL_97:.*]] = constant dense<true> : vector<8xi1>
// CHECK:             %[[VAL_98:.*]] = vector.gather %[[VAL_94]]{{\[}}%[[VAL_95]]], %[[VAL_97]], %[[VAL_96]] : memref<3xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK:             %[[VAL_99:.*]] = get_global_memref @categorical_vec_1 : memref<3xf32>
// CHECK:             %[[VAL_100:.*]] = fptoui %[[VAL_37]] : vector<8xf32> to vector<8xi64>
// CHECK:             %[[VAL_101:.*]] = constant dense<0.000000e+00> : vector<8xf32>
// CHECK:             %[[VAL_102:.*]] = constant dense<true> : vector<8xi1>
// CHECK:             %[[VAL_103:.*]] = vector.gather %[[VAL_99]]{{\[}}%[[VAL_100]]], %[[VAL_102]], %[[VAL_101]] : memref<3xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK:             %[[VAL_104:.*]] = get_global_memref @histogram_vec_0 : memref<2xf32>
// CHECK:             %[[VAL_105:.*]] = fptoui %[[VAL_51]] : vector<8xf32> to vector<8xi64>
// CHECK:             %[[VAL_106:.*]] = constant dense<0.000000e+00> : vector<8xf32>
// CHECK:             %[[VAL_107:.*]] = constant dense<true> : vector<8xi1>
// CHECK:             %[[VAL_108:.*]] = vector.gather %[[VAL_104]]{{\[}}%[[VAL_105]]], %[[VAL_107]], %[[VAL_106]] : memref<2xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK:             %[[VAL_109:.*]] = get_global_memref @histogram_vec_1 : memref<2xf32>
// CHECK:             %[[VAL_110:.*]] = fptoui %[[VAL_65]] : vector<8xf32> to vector<8xi64>
// CHECK:             %[[VAL_111:.*]] = constant dense<0.000000e+00> : vector<8xf32>
// CHECK:             %[[VAL_112:.*]] = constant dense<true> : vector<8xi1>
// CHECK:             %[[VAL_113:.*]] = vector.gather %[[VAL_109]]{{\[}}%[[VAL_110]]], %[[VAL_112]], %[[VAL_111]] : memref<2xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK:             %[[VAL_114:.*]] = constant dense<-5.000000e-01> : vector<8xf32>
// CHECK:             %[[VAL_115:.*]] = constant dense<-0.918938517> : vector<8xf32>
// CHECK:             %[[VAL_116:.*]] = constant dense<5.000000e-01> : vector<8xf32>
// CHECK:             %[[VAL_117:.*]] = subf %[[VAL_79]], %[[VAL_116]] : vector<8xf32>
// CHECK:             %[[VAL_118:.*]] = mulf %[[VAL_117]], %[[VAL_117]] : vector<8xf32>
// CHECK:             %[[VAL_119:.*]] = mulf %[[VAL_118]], %[[VAL_114]] : vector<8xf32>
// CHECK:             %[[VAL_120:.*]] = addf %[[VAL_115]], %[[VAL_119]] : vector<8xf32>
// CHECK:             %[[VAL_121:.*]] = constant dense<-5.000000e+01> : vector<8xf32>
// CHECK:             %[[VAL_122:.*]] = constant dense<1.38364661> : vector<8xf32>
// CHECK:             %[[VAL_123:.*]] = constant dense<2.500000e-01> : vector<8xf32>
// CHECK:             %[[VAL_124:.*]] = subf %[[VAL_93]], %[[VAL_123]] : vector<8xf32>
// CHECK:             %[[VAL_125:.*]] = mulf %[[VAL_124]], %[[VAL_124]] : vector<8xf32>
// CHECK:             %[[VAL_126:.*]] = mulf %[[VAL_125]], %[[VAL_121]] : vector<8xf32>
// CHECK:             %[[VAL_127:.*]] = addf %[[VAL_122]], %[[VAL_126]] : vector<8xf32>
// CHECK:             %[[VAL_128:.*]] = addf %[[VAL_98]], %[[VAL_103]] : vector<8xf32>
// CHECK:             %[[VAL_129:.*]] = addf %[[VAL_128]], %[[VAL_108]] : vector<8xf32>
// CHECK:             %[[VAL_130:.*]] = constant dense<1.000000e-01> : vector<8xf32>
// CHECK:             %[[VAL_131:.*]] = addf %[[VAL_129]], %[[VAL_130]] : vector<8xf32>
// CHECK:             %[[VAL_132:.*]] = addf %[[VAL_113]], %[[VAL_120]] : vector<8xf32>
// CHECK:             %[[VAL_133:.*]] = addf %[[VAL_132]], %[[VAL_127]] : vector<8xf32>
// CHECK:             %[[VAL_134:.*]] = constant dense<1.000000e-01> : vector<8xf32>
// CHECK:             %[[VAL_135:.*]] = addf %[[VAL_133]], %[[VAL_134]] : vector<8xf32>
// CHECK:             %[[VAL_136:.*]] = cmpf ogt, %[[VAL_131]], %[[VAL_135]] : vector<8xf32>
// CHECK:             %[[VAL_137:.*]] = select %[[VAL_136]], %[[VAL_131]], %[[VAL_135]] : vector<8xi1>, vector<8xf32>
// CHECK:             %[[VAL_138:.*]] = select %[[VAL_136]], %[[VAL_135]], %[[VAL_131]] : vector<8xi1>, vector<8xf32>
// CHECK:             %[[VAL_139:.*]] = subf %[[VAL_138]], %[[VAL_137]] : vector<8xf32>
// CHECK:             %[[VAL_140:.*]] = math.exp %[[VAL_139]] : vector<8xf32>
// CHECK:             %[[VAL_141:.*]] = constant dense<1.000000e+00> : vector<8xf32>
// CHECK:             %[[VAL_142:.*]] = addf %[[VAL_141]], %[[VAL_140]] : vector<8xf32>
// CHECK:             %[[VAL_143:.*]] = math.log %[[VAL_142]] : vector<8xf32>
// CHECK:             %[[VAL_144:.*]] = addf %[[VAL_137]], %[[VAL_143]] : vector<8xf32>
// CHECK:             vector.transfer_write %[[VAL_144]], %[[VAL_1]]{{\[}}%[[VAL_9]]] : vector<8xf32>, memref<?xf32>
// CHECK:           }
// CHECK:           %[[VAL_145:.*]] = constant 1 : index
// CHECK:           scf.for %[[VAL_146:.*]] = %[[VAL_6]] to %[[VAL_3]] step %[[VAL_145]] {
// CHECK:             %[[VAL_147:.*]] = constant 0 : index
// CHECK:             %[[VAL_148:.*]] = load %[[VAL_0]]{{\[}}%[[VAL_146]], %[[VAL_147]]] : memref<?x6xf32>
// CHECK:             %[[VAL_149:.*]] = constant 1 : index
// CHECK:             %[[VAL_150:.*]] = load %[[VAL_0]]{{\[}}%[[VAL_146]], %[[VAL_149]]] : memref<?x6xf32>
// CHECK:             %[[VAL_151:.*]] = constant 2 : index
// CHECK:             %[[VAL_152:.*]] = load %[[VAL_0]]{{\[}}%[[VAL_146]], %[[VAL_151]]] : memref<?x6xf32>
// CHECK:             %[[VAL_153:.*]] = constant 3 : index
// CHECK:             %[[VAL_154:.*]] = load %[[VAL_0]]{{\[}}%[[VAL_146]], %[[VAL_153]]] : memref<?x6xf32>
// CHECK:             %[[VAL_155:.*]] = constant 4 : index
// CHECK:             %[[VAL_156:.*]] = load %[[VAL_0]]{{\[}}%[[VAL_146]], %[[VAL_155]]] : memref<?x6xf32>
// CHECK:             %[[VAL_157:.*]] = constant 5 : index
// CHECK:             %[[VAL_158:.*]] = load %[[VAL_0]]{{\[}}%[[VAL_146]], %[[VAL_157]]] : memref<?x6xf32>
// CHECK:             %[[VAL_159:.*]] = get_global_memref @categorical_0 : memref<3xf32>
// CHECK:             %[[VAL_160:.*]] = fptoui %[[VAL_148]] : f32 to i64
// CHECK:             %[[VAL_161:.*]] = index_cast %[[VAL_160]] : i64 to index
// CHECK:             %[[VAL_162:.*]] = load %[[VAL_159]]{{\[}}%[[VAL_161]]] : memref<3xf32>
// CHECK:             %[[VAL_163:.*]] = get_global_memref @categorical_1 : memref<3xf32>
// CHECK:             %[[VAL_164:.*]] = fptoui %[[VAL_150]] : f32 to i64
// CHECK:             %[[VAL_165:.*]] = index_cast %[[VAL_164]] : i64 to index
// CHECK:             %[[VAL_166:.*]] = load %[[VAL_163]]{{\[}}%[[VAL_165]]] : memref<3xf32>
// CHECK:             %[[VAL_167:.*]] = get_global_memref @histogram_0 : memref<2xf32>
// CHECK:             %[[VAL_168:.*]] = fptoui %[[VAL_152]] : f32 to i64
// CHECK:             %[[VAL_169:.*]] = index_cast %[[VAL_168]] : i64 to index
// CHECK:             %[[VAL_170:.*]] = load %[[VAL_167]]{{\[}}%[[VAL_169]]] : memref<2xf32>
// CHECK:             %[[VAL_171:.*]] = get_global_memref @histogram_1 : memref<2xf32>
// CHECK:             %[[VAL_172:.*]] = fptoui %[[VAL_154]] : f32 to i64
// CHECK:             %[[VAL_173:.*]] = index_cast %[[VAL_172]] : i64 to index
// CHECK:             %[[VAL_174:.*]] = load %[[VAL_171]]{{\[}}%[[VAL_173]]] : memref<2xf32>
// CHECK:             %[[VAL_175:.*]] = constant -5.000000e-01 : f32
// CHECK:             %[[VAL_176:.*]] = constant -0.918938517 : f32
// CHECK:             %[[VAL_177:.*]] = constant 5.000000e-01 : f32
// CHECK:             %[[VAL_178:.*]] = subf %[[VAL_156]], %[[VAL_177]] : f32
// CHECK:             %[[VAL_179:.*]] = mulf %[[VAL_178]], %[[VAL_178]] : f32
// CHECK:             %[[VAL_180:.*]] = mulf %[[VAL_179]], %[[VAL_175]] : f32
// CHECK:             %[[VAL_181:.*]] = addf %[[VAL_176]], %[[VAL_180]] : f32
// CHECK:             %[[VAL_182:.*]] = constant -5.000000e+01 : f32
// CHECK:             %[[VAL_183:.*]] = constant 1.38364661 : f32
// CHECK:             %[[VAL_184:.*]] = constant 2.500000e-01 : f32
// CHECK:             %[[VAL_185:.*]] = subf %[[VAL_158]], %[[VAL_184]] : f32
// CHECK:             %[[VAL_186:.*]] = mulf %[[VAL_185]], %[[VAL_185]] : f32
// CHECK:             %[[VAL_187:.*]] = mulf %[[VAL_186]], %[[VAL_182]] : f32
// CHECK:             %[[VAL_188:.*]] = addf %[[VAL_183]], %[[VAL_187]] : f32
// CHECK:             %[[VAL_189:.*]] = addf %[[VAL_162]], %[[VAL_166]] : f32
// CHECK:             %[[VAL_190:.*]] = addf %[[VAL_189]], %[[VAL_170]] : f32
// CHECK:             %[[VAL_191:.*]] = constant 1.000000e-01 : f32
// CHECK:             %[[VAL_192:.*]] = addf %[[VAL_190]], %[[VAL_191]] : f32
// CHECK:             %[[VAL_193:.*]] = addf %[[VAL_174]], %[[VAL_181]] : f32
// CHECK:             %[[VAL_194:.*]] = addf %[[VAL_193]], %[[VAL_188]] : f32
// CHECK:             %[[VAL_195:.*]] = constant 1.000000e-01 : f32
// CHECK:             %[[VAL_196:.*]] = addf %[[VAL_194]], %[[VAL_195]] : f32
// CHECK:             %[[VAL_197:.*]] = cmpf ogt, %[[VAL_192]], %[[VAL_196]] : f32
// CHECK:             %[[VAL_198:.*]] = select %[[VAL_197]], %[[VAL_192]], %[[VAL_196]] : f32
// CHECK:             %[[VAL_199:.*]] = select %[[VAL_197]], %[[VAL_196]], %[[VAL_192]] : f32
// CHECK:             %[[VAL_200:.*]] = subf %[[VAL_199]], %[[VAL_198]] : f32
// CHECK:             %[[VAL_201:.*]] = math.exp %[[VAL_200]] : f32
// CHECK:             %[[VAL_202:.*]] = constant 1.000000e+00 : f32
// CHECK:             %[[VAL_203:.*]] = addf %[[VAL_202]], %[[VAL_201]] : f32
// CHECK:             %[[VAL_204:.*]] = math.log %[[VAL_203]] : f32
// CHECK:             %[[VAL_205:.*]] = addf %[[VAL_198]], %[[VAL_204]] : f32
// CHECK:             store %[[VAL_205]], %[[VAL_1]]{{\[}}%[[VAL_146]]] : memref<?xf32>
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func @spn_vector(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<?x6xf32>,
// CHECK-SAME:                     %[[VAL_1:.*]]: memref<?xf32>) {
// CHECK:           %[[VAL_2:.*]] = constant 0 : index
// CHECK:           %[[VAL_3:.*]] = dim %[[VAL_0]], %[[VAL_2]] : memref<?x6xf32>
// CHECK:           %[[VAL_4:.*]] = alloc(%[[VAL_3]]) : memref<?xf32>
// CHECK:           call @vec_task_0(%[[VAL_0]], %[[VAL_4]]) : (memref<?x6xf32>, memref<?xf32>) -> ()
// CHECK:           %[[VAL_5:.*]] = tensor_load %[[VAL_4]] : memref<?xf32>
// CHECK:           %[[VAL_6:.*]] = tensor_to_memref %[[VAL_5]] : memref<?xf32>
// CHECK:           linalg.copy(%[[VAL_6]], %[[VAL_1]]) : memref<?xf32>, memref<?xf32>
// CHECK:           return
// CHECK:         }
