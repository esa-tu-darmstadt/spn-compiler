// RUN: %optcall --vectorize-lospn-nodes %s | FileCheck %s

module  {
  func @vec_task_0(%arg0: memref<?x6xf32>, %arg1: memref<?xf32>) {
    %c0 = constant 0 : index
    %0 = memref.dim %arg0, %c0 : memref<?x6xf32>
    %c4 = constant 4 : index
    %1 = remi_unsigned %0, %c4 : index
    %2 = subi %0, %1 : index
    %c0_0 = constant 0 : index
    %c4_1 = constant 4 : index
    scf.for %arg2 = %c0_0 to %2 step %c4_1 {
      %3 = "lo_spn.batch_read"(%arg0, %arg2) {sampleIndex = 0 : ui32, vector_width = 8 : i32} : (memref<?x6xf32>, index) -> f32
      %4 = "lo_spn.batch_read"(%arg0, %arg2) {sampleIndex = 1 : ui32, vector_width = 8 : i32} : (memref<?x6xf32>, index) -> f32
      %5 = "lo_spn.batch_read"(%arg0, %arg2) {sampleIndex = 2 : ui32, vector_width = 8 : i32} : (memref<?x6xf32>, index) -> f32
      %6 = "lo_spn.batch_read"(%arg0, %arg2) {sampleIndex = 3 : ui32, vector_width = 8 : i32} : (memref<?x6xf32>, index) -> f32
      %7 = "lo_spn.batch_read"(%arg0, %arg2) {sampleIndex = 4 : ui32, vector_width = 8 : i32} : (memref<?x6xf32>, index) -> f32
      %8 = "lo_spn.batch_read"(%arg0, %arg2) {sampleIndex = 5 : ui32, vector_width = 8 : i32} : (memref<?x6xf32>, index) -> f32
      %9 = "lo_spn.categorical"(%3) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false, vector_width = 8 : i32} : (f32) -> !lo_spn.log<f32>
      %10 = "lo_spn.categorical"(%4) {probabilities = [2.500000e-01, 6.250000e-01, 1.250000e-01], supportMarginal = false, vector_width = 8 : i32} : (f32) -> !lo_spn.log<f32>
      %11 = "lo_spn.histogram"(%5) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}], supportMarginal = false, vector_width = 8 : i32} : (f32) -> !lo_spn.log<f32>
      %12 = "lo_spn.histogram"(%6) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 4.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 5.500000e-01 : f64}], supportMarginal = false, vector_width = 8 : i32} : (f32) -> !lo_spn.log<f32>
      %13 = "lo_spn.gaussian"(%7) {mean = 5.000000e-01 : f64, stddev = 1.000000e+00 : f64, supportMarginal = false, vector_width = 8 : i32} : (f32) -> !lo_spn.log<f32>
      %14 = "lo_spn.gaussian"(%8) {mean = 2.500000e-01 : f64, stddev = 1.000000e-01 : f64, supportMarginal = false, vector_width = 8 : i32} : (f32) -> !lo_spn.log<f32>
      %15 = "lo_spn.mul"(%9, %10) {vector_width = 8 : i32} : (!lo_spn.log<f32>, !lo_spn.log<f32>) -> !lo_spn.log<f32>
      %16 = "lo_spn.mul"(%15, %11) {vector_width = 8 : i32} : (!lo_spn.log<f32>, !lo_spn.log<f32>) -> !lo_spn.log<f32>
      %17 = "lo_spn.constant"() {type = !lo_spn.log<f32>, value = 1.000000e-01 : f64, vector_width = 8 : i32} : () -> !lo_spn.log<f32>
      %18 = "lo_spn.mul"(%16, %17) {vector_width = 8 : i32} : (!lo_spn.log<f32>, !lo_spn.log<f32>) -> !lo_spn.log<f32>
      %19 = "lo_spn.mul"(%12, %13) {vector_width = 8 : i32} : (!lo_spn.log<f32>, !lo_spn.log<f32>) -> !lo_spn.log<f32>
      %20 = "lo_spn.mul"(%19, %14) {vector_width = 8 : i32} : (!lo_spn.log<f32>, !lo_spn.log<f32>) -> !lo_spn.log<f32>
      %21 = "lo_spn.constant"() {type = !lo_spn.log<f32>, value = 1.000000e-01 : f64, vector_width = 8 : i32} : () -> !lo_spn.log<f32>
      %22 = "lo_spn.mul"(%20, %21) {vector_width = 8 : i32} : (!lo_spn.log<f32>, !lo_spn.log<f32>) -> !lo_spn.log<f32>
      %23 = "lo_spn.add"(%18, %22) {vector_width = 8 : i32} : (!lo_spn.log<f32>, !lo_spn.log<f32>) -> !lo_spn.log<f32>
      %24 = "lo_spn.strip_log"(%23) {target = f32, vector_width = 8 : i32} : (!lo_spn.log<f32>) -> f32
      "lo_spn.batch_write"(%24, %arg1, %arg2) {vector_width = 8 : i32} : (f32, memref<?xf32>, index) -> ()
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
    %0 = memref.dim %arg0, %c0 : memref<?x6xf32>
    %1 = memref.alloc(%0) : memref<?xf32>
    call @vec_task_0(%arg0, %1) : (memref<?x6xf32>, memref<?xf32>) -> ()
    %2 = memref.tensor_load %1 : memref<?xf32>
    %3 = memref.buffer_cast %2 : memref<?xf32>
    "lo_spn.copy"(%3, %arg1) : (memref<?xf32>, memref<?xf32>) -> ()
    "lo_spn.return"() : () -> ()
  }
}

// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

// CHECK-LABEL:   memref.global "private" constant @histogram_vec_1 : memref<2xf32> = dense<[-0.79850769, -0.597836971]>
// CHECK:         memref.global "private" constant @histogram_vec_0 : memref<2xf32> = dense<[-1.38629436, -0.287682086]>
// CHECK:         memref.global "private" constant @categorical_vec_1 : memref<3xf32> = dense<[-1.38629436, -0.470003635, -2.07944155]>
// CHECK:         memref.global "private" constant @categorical_vec_0 : memref<3xf32> = dense<[-1.04982209, -0.597836971, -2.30258512]>

// CHECK-LABEL:   func @vec_task_0(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<?x6xf32>,
// CHECK-SAME:                     %[[VAL_1:.*]]: memref<?xf32>) {
// CHECK:           %[[VAL_2:.*]] = constant 0 : index
// CHECK:           %[[VAL_3:.*]] = memref.dim %[[VAL_0]], %[[VAL_2]] : memref<?x6xf32>
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
// CHECK:             %[[VAL_19:.*]] = memref.dim %[[VAL_0]], %[[VAL_18]] : memref<?x6xf32>
// CHECK:             %[[VAL_20:.*]] = constant 6 : index
// CHECK:             %[[VAL_21:.*]] = muli %[[VAL_19]], %[[VAL_20]] : index
// CHECK:             %[[VAL_22:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_21]]], strides: [1] : memref<?x6xf32> to memref<?xf32>
// CHECK:             %[[VAL_23:.*]] = constant 0 : index
// CHECK:             %[[VAL_24:.*]] = vector.gather %[[VAL_22]]{{\[}}%[[VAL_23]]] {{\[}}%[[VAL_15]]], %[[VAL_17]], %[[VAL_16]] : memref<?xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK:             %[[VAL_25:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_26:.*]] = vector.broadcast %[[VAL_25]] : i64 to vector<8xi64>
// CHECK:             %[[VAL_27:.*]] = constant dense<[1, 7, 13, 19, 25, 31, 37, 43]> : vector<8xi64>
// CHECK:             %[[VAL_28:.*]] = constant dense<6> : vector<8xi64>
// CHECK:             %[[VAL_29:.*]] = muli %[[VAL_26]], %[[VAL_28]] : vector<8xi64>
// CHECK:             %[[VAL_30:.*]] = addi %[[VAL_29]], %[[VAL_27]] : vector<8xi64>
// CHECK:             %[[VAL_31:.*]] = constant dense<0.000000e+00> : vector<8xf32>
// CHECK:             %[[VAL_32:.*]] = constant dense<true> : vector<8xi1>
// CHECK:             %[[VAL_33:.*]] = constant 0 : index
// CHECK:             %[[VAL_34:.*]] = memref.dim %[[VAL_0]], %[[VAL_33]] : memref<?x6xf32>
// CHECK:             %[[VAL_35:.*]] = constant 6 : index
// CHECK:             %[[VAL_36:.*]] = muli %[[VAL_34]], %[[VAL_35]] : index
// CHECK:             %[[VAL_37:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_36]]], strides: [1] : memref<?x6xf32> to memref<?xf32>
// CHECK:             %[[VAL_38:.*]] = constant 0 : index
// CHECK:             %[[VAL_39:.*]] = vector.gather %[[VAL_37]]{{\[}}%[[VAL_38]]] {{\[}}%[[VAL_30]]], %[[VAL_32]], %[[VAL_31]] : memref<?xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK:             %[[VAL_40:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_41:.*]] = vector.broadcast %[[VAL_40]] : i64 to vector<8xi64>
// CHECK:             %[[VAL_42:.*]] = constant dense<[2, 8, 14, 20, 26, 32, 38, 44]> : vector<8xi64>
// CHECK:             %[[VAL_43:.*]] = constant dense<6> : vector<8xi64>
// CHECK:             %[[VAL_44:.*]] = muli %[[VAL_41]], %[[VAL_43]] : vector<8xi64>
// CHECK:             %[[VAL_45:.*]] = addi %[[VAL_44]], %[[VAL_42]] : vector<8xi64>
// CHECK:             %[[VAL_46:.*]] = constant dense<0.000000e+00> : vector<8xf32>
// CHECK:             %[[VAL_47:.*]] = constant dense<true> : vector<8xi1>
// CHECK:             %[[VAL_48:.*]] = constant 0 : index
// CHECK:             %[[VAL_49:.*]] = memref.dim %[[VAL_0]], %[[VAL_48]] : memref<?x6xf32>
// CHECK:             %[[VAL_50:.*]] = constant 6 : index
// CHECK:             %[[VAL_51:.*]] = muli %[[VAL_49]], %[[VAL_50]] : index
// CHECK:             %[[VAL_52:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_51]]], strides: [1] : memref<?x6xf32> to memref<?xf32>
// CHECK:             %[[VAL_53:.*]] = constant 0 : index
// CHECK:             %[[VAL_54:.*]] = vector.gather %[[VAL_52]]{{\[}}%[[VAL_53]]] {{\[}}%[[VAL_45]]], %[[VAL_47]], %[[VAL_46]] : memref<?xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK:             %[[VAL_55:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_56:.*]] = vector.broadcast %[[VAL_55]] : i64 to vector<8xi64>
// CHECK:             %[[VAL_57:.*]] = constant dense<[3, 9, 15, 21, 27, 33, 39, 45]> : vector<8xi64>
// CHECK:             %[[VAL_58:.*]] = constant dense<6> : vector<8xi64>
// CHECK:             %[[VAL_59:.*]] = muli %[[VAL_56]], %[[VAL_58]] : vector<8xi64>
// CHECK:             %[[VAL_60:.*]] = addi %[[VAL_59]], %[[VAL_57]] : vector<8xi64>
// CHECK:             %[[VAL_61:.*]] = constant dense<0.000000e+00> : vector<8xf32>
// CHECK:             %[[VAL_62:.*]] = constant dense<true> : vector<8xi1>
// CHECK:             %[[VAL_63:.*]] = constant 0 : index
// CHECK:             %[[VAL_64:.*]] = memref.dim %[[VAL_0]], %[[VAL_63]] : memref<?x6xf32>
// CHECK:             %[[VAL_65:.*]] = constant 6 : index
// CHECK:             %[[VAL_66:.*]] = muli %[[VAL_64]], %[[VAL_65]] : index
// CHECK:             %[[VAL_67:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_66]]], strides: [1] : memref<?x6xf32> to memref<?xf32>
// CHECK:             %[[VAL_68:.*]] = constant 0 : index
// CHECK:             %[[VAL_69:.*]] = vector.gather %[[VAL_67]]{{\[}}%[[VAL_68]]] {{\[}}%[[VAL_60]]], %[[VAL_62]], %[[VAL_61]] : memref<?xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK:             %[[VAL_70:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_71:.*]] = vector.broadcast %[[VAL_70]] : i64 to vector<8xi64>
// CHECK:             %[[VAL_72:.*]] = constant dense<[4, 10, 16, 22, 28, 34, 40, 46]> : vector<8xi64>
// CHECK:             %[[VAL_73:.*]] = constant dense<6> : vector<8xi64>
// CHECK:             %[[VAL_74:.*]] = muli %[[VAL_71]], %[[VAL_73]] : vector<8xi64>
// CHECK:             %[[VAL_75:.*]] = addi %[[VAL_74]], %[[VAL_72]] : vector<8xi64>
// CHECK:             %[[VAL_76:.*]] = constant dense<0.000000e+00> : vector<8xf32>
// CHECK:             %[[VAL_77:.*]] = constant dense<true> : vector<8xi1>
// CHECK:             %[[VAL_78:.*]] = constant 0 : index
// CHECK:             %[[VAL_79:.*]] = memref.dim %[[VAL_0]], %[[VAL_78]] : memref<?x6xf32>
// CHECK:             %[[VAL_80:.*]] = constant 6 : index
// CHECK:             %[[VAL_81:.*]] = muli %[[VAL_79]], %[[VAL_80]] : index
// CHECK:             %[[VAL_82:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_81]]], strides: [1] : memref<?x6xf32> to memref<?xf32>
// CHECK:             %[[VAL_83:.*]] = constant 0 : index
// CHECK:             %[[VAL_84:.*]] = vector.gather %[[VAL_82]]{{\[}}%[[VAL_83]]] {{\[}}%[[VAL_75]]], %[[VAL_77]], %[[VAL_76]] : memref<?xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK:             %[[VAL_85:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_86:.*]] = vector.broadcast %[[VAL_85]] : i64 to vector<8xi64>
// CHECK:             %[[VAL_87:.*]] = constant dense<[5, 11, 17, 23, 29, 35, 41, 47]> : vector<8xi64>
// CHECK:             %[[VAL_88:.*]] = constant dense<6> : vector<8xi64>
// CHECK:             %[[VAL_89:.*]] = muli %[[VAL_86]], %[[VAL_88]] : vector<8xi64>
// CHECK:             %[[VAL_90:.*]] = addi %[[VAL_89]], %[[VAL_87]] : vector<8xi64>
// CHECK:             %[[VAL_91:.*]] = constant dense<0.000000e+00> : vector<8xf32>
// CHECK:             %[[VAL_92:.*]] = constant dense<true> : vector<8xi1>
// CHECK:             %[[VAL_93:.*]] = constant 0 : index
// CHECK:             %[[VAL_94:.*]] = memref.dim %[[VAL_0]], %[[VAL_93]] : memref<?x6xf32>
// CHECK:             %[[VAL_95:.*]] = constant 6 : index
// CHECK:             %[[VAL_96:.*]] = muli %[[VAL_94]], %[[VAL_95]] : index
// CHECK:             %[[VAL_97:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_96]]], strides: [1] : memref<?x6xf32> to memref<?xf32>
// CHECK:             %[[VAL_98:.*]] = constant 0 : index
// CHECK:             %[[VAL_99:.*]] = vector.gather %[[VAL_97]]{{\[}}%[[VAL_98]]] {{\[}}%[[VAL_90]]], %[[VAL_92]], %[[VAL_91]] : memref<?xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK:             %[[VAL_100:.*]] = memref.get_global @categorical_vec_0 : memref<3xf32>
// CHECK:             %[[VAL_101:.*]] = fptoui %[[VAL_24]] : vector<8xf32> to vector<8xi64>
// CHECK:             %[[VAL_102:.*]] = constant dense<0xFF800000> : vector<8xf32>
// CHECK:             %[[VAL_103:.*]] = constant dense<0> : vector<8xi64>
// CHECK:             %[[VAL_104:.*]] = constant dense<3> : vector<8xi64>
// CHECK:             %[[VAL_105:.*]] = cmpi sge, %[[VAL_101]], %[[VAL_103]] : vector<8xi64>
// CHECK:             %[[VAL_106:.*]] = cmpi slt, %[[VAL_101]], %[[VAL_104]] : vector<8xi64>
// CHECK:             %[[VAL_107:.*]] = and %[[VAL_105]], %[[VAL_106]] : vector<8xi1>
// CHECK:             %[[VAL_108:.*]] = constant dense<true> : vector<8xi1>
// CHECK:             %[[VAL_109:.*]] = constant dense<false> : vector<8xi1>
// CHECK:             %[[VAL_110:.*]] = select %[[VAL_107]], %[[VAL_108]], %[[VAL_109]] : vector<8xi1>, vector<8xi1>
// CHECK:             %[[VAL_111:.*]] = constant 0 : index
// CHECK:             %[[VAL_112:.*]] = vector.gather %[[VAL_100]]{{\[}}%[[VAL_111]]] {{\[}}%[[VAL_101]]], %[[VAL_110]], %[[VAL_102]] : memref<3xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK:             %[[VAL_113:.*]] = memref.get_global @categorical_vec_1 : memref<3xf32>
// CHECK:             %[[VAL_114:.*]] = fptoui %[[VAL_39]] : vector<8xf32> to vector<8xi64>
// CHECK:             %[[VAL_115:.*]] = constant dense<0xFF800000> : vector<8xf32>
// CHECK:             %[[VAL_116:.*]] = constant dense<0> : vector<8xi64>
// CHECK:             %[[VAL_117:.*]] = constant dense<3> : vector<8xi64>
// CHECK:             %[[VAL_118:.*]] = cmpi sge, %[[VAL_114]], %[[VAL_116]] : vector<8xi64>
// CHECK:             %[[VAL_119:.*]] = cmpi slt, %[[VAL_114]], %[[VAL_117]] : vector<8xi64>
// CHECK:             %[[VAL_120:.*]] = and %[[VAL_118]], %[[VAL_119]] : vector<8xi1>
// CHECK:             %[[VAL_121:.*]] = constant dense<true> : vector<8xi1>
// CHECK:             %[[VAL_122:.*]] = constant dense<false> : vector<8xi1>
// CHECK:             %[[VAL_123:.*]] = select %[[VAL_120]], %[[VAL_121]], %[[VAL_122]] : vector<8xi1>, vector<8xi1>
// CHECK:             %[[VAL_124:.*]] = constant 0 : index
// CHECK:             %[[VAL_125:.*]] = vector.gather %[[VAL_113]]{{\[}}%[[VAL_124]]] {{\[}}%[[VAL_114]]], %[[VAL_123]], %[[VAL_115]] : memref<3xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK:             %[[VAL_126:.*]] = memref.get_global @histogram_vec_0 : memref<2xf32>
// CHECK:             %[[VAL_127:.*]] = fptoui %[[VAL_54]] : vector<8xf32> to vector<8xi64>
// CHECK:             %[[VAL_128:.*]] = constant dense<0xFF800000> : vector<8xf32>
// CHECK:             %[[VAL_129:.*]] = constant dense<0> : vector<8xi64>
// CHECK:             %[[VAL_130:.*]] = constant dense<2> : vector<8xi64>
// CHECK:             %[[VAL_131:.*]] = cmpi sge, %[[VAL_127]], %[[VAL_129]] : vector<8xi64>
// CHECK:             %[[VAL_132:.*]] = cmpi slt, %[[VAL_127]], %[[VAL_130]] : vector<8xi64>
// CHECK:             %[[VAL_133:.*]] = and %[[VAL_131]], %[[VAL_132]] : vector<8xi1>
// CHECK:             %[[VAL_134:.*]] = constant dense<true> : vector<8xi1>
// CHECK:             %[[VAL_135:.*]] = constant dense<false> : vector<8xi1>
// CHECK:             %[[VAL_136:.*]] = select %[[VAL_133]], %[[VAL_134]], %[[VAL_135]] : vector<8xi1>, vector<8xi1>
// CHECK:             %[[VAL_137:.*]] = constant 0 : index
// CHECK:             %[[VAL_138:.*]] = vector.gather %[[VAL_126]]{{\[}}%[[VAL_137]]] {{\[}}%[[VAL_127]]], %[[VAL_136]], %[[VAL_128]] : memref<2xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK:             %[[VAL_139:.*]] = memref.get_global @histogram_vec_1 : memref<2xf32>
// CHECK:             %[[VAL_140:.*]] = fptoui %[[VAL_69]] : vector<8xf32> to vector<8xi64>
// CHECK:             %[[VAL_141:.*]] = constant dense<0xFF800000> : vector<8xf32>
// CHECK:             %[[VAL_142:.*]] = constant dense<0> : vector<8xi64>
// CHECK:             %[[VAL_143:.*]] = constant dense<2> : vector<8xi64>
// CHECK:             %[[VAL_144:.*]] = cmpi sge, %[[VAL_140]], %[[VAL_142]] : vector<8xi64>
// CHECK:             %[[VAL_145:.*]] = cmpi slt, %[[VAL_140]], %[[VAL_143]] : vector<8xi64>
// CHECK:             %[[VAL_146:.*]] = and %[[VAL_144]], %[[VAL_145]] : vector<8xi1>
// CHECK:             %[[VAL_147:.*]] = constant dense<true> : vector<8xi1>
// CHECK:             %[[VAL_148:.*]] = constant dense<false> : vector<8xi1>
// CHECK:             %[[VAL_149:.*]] = select %[[VAL_146]], %[[VAL_147]], %[[VAL_148]] : vector<8xi1>, vector<8xi1>
// CHECK:             %[[VAL_150:.*]] = constant 0 : index
// CHECK:             %[[VAL_151:.*]] = vector.gather %[[VAL_139]]{{\[}}%[[VAL_150]]] {{\[}}%[[VAL_140]]], %[[VAL_149]], %[[VAL_141]] : memref<2xf32>, vector<8xi64>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK:             %[[VAL_152:.*]] = constant dense<-5.000000e-01> : vector<8xf32>
// CHECK:             %[[VAL_153:.*]] = constant dense<-0.918938517> : vector<8xf32>
// CHECK:             %[[VAL_154:.*]] = constant dense<5.000000e-01> : vector<8xf32>
// CHECK:             %[[VAL_155:.*]] = subf %[[VAL_84]], %[[VAL_154]] : vector<8xf32>
// CHECK:             %[[VAL_156:.*]] = mulf %[[VAL_155]], %[[VAL_155]] : vector<8xf32>
// CHECK:             %[[VAL_157:.*]] = mulf %[[VAL_156]], %[[VAL_152]] : vector<8xf32>
// CHECK:             %[[VAL_158:.*]] = addf %[[VAL_153]], %[[VAL_157]] : vector<8xf32>
// CHECK:             %[[VAL_159:.*]] = constant dense<-5.000000e+01> : vector<8xf32>
// CHECK:             %[[VAL_160:.*]] = constant dense<1.38364661> : vector<8xf32>
// CHECK:             %[[VAL_161:.*]] = constant dense<2.500000e-01> : vector<8xf32>
// CHECK:             %[[VAL_162:.*]] = subf %[[VAL_99]], %[[VAL_161]] : vector<8xf32>
// CHECK:             %[[VAL_163:.*]] = mulf %[[VAL_162]], %[[VAL_162]] : vector<8xf32>
// CHECK:             %[[VAL_164:.*]] = mulf %[[VAL_163]], %[[VAL_159]] : vector<8xf32>
// CHECK:             %[[VAL_165:.*]] = addf %[[VAL_160]], %[[VAL_164]] : vector<8xf32>
// CHECK:             %[[VAL_166:.*]] = addf %[[VAL_112]], %[[VAL_125]] : vector<8xf32>
// CHECK:             %[[VAL_167:.*]] = addf %[[VAL_166]], %[[VAL_138]] : vector<8xf32>
// CHECK:             %[[VAL_168:.*]] = constant dense<1.000000e-01> : vector<8xf32>
// CHECK:             %[[VAL_169:.*]] = addf %[[VAL_167]], %[[VAL_168]] : vector<8xf32>
// CHECK:             %[[VAL_170:.*]] = addf %[[VAL_151]], %[[VAL_158]] : vector<8xf32>
// CHECK:             %[[VAL_171:.*]] = addf %[[VAL_170]], %[[VAL_165]] : vector<8xf32>
// CHECK:             %[[VAL_172:.*]] = constant dense<1.000000e-01> : vector<8xf32>
// CHECK:             %[[VAL_173:.*]] = addf %[[VAL_171]], %[[VAL_172]] : vector<8xf32>
// CHECK:             %[[VAL_174:.*]] = cmpf ogt, %[[VAL_169]], %[[VAL_173]] : vector<8xf32>
// CHECK:             %[[VAL_175:.*]] = select %[[VAL_174]], %[[VAL_169]], %[[VAL_173]] : vector<8xi1>, vector<8xf32>
// CHECK:             %[[VAL_176:.*]] = select %[[VAL_174]], %[[VAL_173]], %[[VAL_169]] : vector<8xi1>, vector<8xf32>
// CHECK:             %[[VAL_177:.*]] = subf %[[VAL_176]], %[[VAL_175]] : vector<8xf32>
// CHECK:             %[[VAL_178:.*]] = math.exp %[[VAL_177]] : vector<8xf32>
// CHECK:             %[[VAL_179:.*]] = math.log1p %[[VAL_178]] : vector<8xf32>
// CHECK:             %[[VAL_180:.*]] = addf %[[VAL_175]], %[[VAL_179]] : vector<8xf32>
// CHECK:             vector.transfer_write %[[VAL_180]], %[[VAL_1]]{{\[}}%[[VAL_9]]] : vector<8xf32>, memref<?xf32>
// CHECK:           }
// CHECK:           %[[VAL_181:.*]] = constant 1 : index
// CHECK:           scf.for %[[VAL_182:.*]] = %[[VAL_6]] to %[[VAL_3]] step %[[VAL_181]] {
// CHECK:             %[[VAL_183:.*]] = "lo_spn.batch_read"(%[[VAL_0]], %[[VAL_182]]) {sampleIndex = 0 : ui32} : (memref<?x6xf32>, index) -> f32
// CHECK:             %[[VAL_184:.*]] = "lo_spn.batch_read"(%[[VAL_0]], %[[VAL_182]]) {sampleIndex = 1 : ui32} : (memref<?x6xf32>, index) -> f32
// CHECK:             %[[VAL_185:.*]] = "lo_spn.batch_read"(%[[VAL_0]], %[[VAL_182]]) {sampleIndex = 2 : ui32} : (memref<?x6xf32>, index) -> f32
// CHECK:             %[[VAL_186:.*]] = "lo_spn.batch_read"(%[[VAL_0]], %[[VAL_182]]) {sampleIndex = 3 : ui32} : (memref<?x6xf32>, index) -> f32
// CHECK:             %[[VAL_187:.*]] = "lo_spn.batch_read"(%[[VAL_0]], %[[VAL_182]]) {sampleIndex = 4 : ui32} : (memref<?x6xf32>, index) -> f32
// CHECK:             %[[VAL_188:.*]] = "lo_spn.batch_read"(%[[VAL_0]], %[[VAL_182]]) {sampleIndex = 5 : ui32} : (memref<?x6xf32>, index) -> f32
// CHECK:             %[[VAL_189:.*]] = "lo_spn.categorical"(%[[VAL_183]]) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false} : (f32) -> !lo_spn.log<f32>
// CHECK:             %[[VAL_190:.*]] = "lo_spn.categorical"(%[[VAL_184]]) {probabilities = [2.500000e-01, 6.250000e-01, 1.250000e-01], supportMarginal = false} : (f32) -> !lo_spn.log<f32>
// CHECK:             %[[VAL_191:.*]] = "lo_spn.histogram"(%[[VAL_185]]) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}], supportMarginal = false} : (f32) -> !lo_spn.log<f32>
// CHECK:             %[[VAL_192:.*]] = "lo_spn.histogram"(%[[VAL_186]]) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 4.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 5.500000e-01 : f64}], supportMarginal = false} : (f32) -> !lo_spn.log<f32>
// CHECK:             %[[VAL_193:.*]] = "lo_spn.gaussian"(%[[VAL_187]]) {mean = 5.000000e-01 : f64, stddev = 1.000000e+00 : f64, supportMarginal = false} : (f32) -> !lo_spn.log<f32>
// CHECK:             %[[VAL_194:.*]] = "lo_spn.gaussian"(%[[VAL_188]]) {mean = 2.500000e-01 : f64, stddev = 1.000000e-01 : f64, supportMarginal = false} : (f32) -> !lo_spn.log<f32>
// CHECK:             %[[VAL_195:.*]] = "lo_spn.mul"(%[[VAL_189]], %[[VAL_190]]) : (!lo_spn.log<f32>, !lo_spn.log<f32>) -> !lo_spn.log<f32>
// CHECK:             %[[VAL_196:.*]] = "lo_spn.mul"(%[[VAL_195]], %[[VAL_191]]) : (!lo_spn.log<f32>, !lo_spn.log<f32>) -> !lo_spn.log<f32>
// CHECK:             %[[VAL_197:.*]] = "lo_spn.constant"() {type = !lo_spn.log<f32>, value = 1.000000e-01 : f64} : () -> !lo_spn.log<f32>
// CHECK:             %[[VAL_198:.*]] = "lo_spn.mul"(%[[VAL_196]], %[[VAL_197]]) : (!lo_spn.log<f32>, !lo_spn.log<f32>) -> !lo_spn.log<f32>
// CHECK:             %[[VAL_199:.*]] = "lo_spn.mul"(%[[VAL_192]], %[[VAL_193]]) : (!lo_spn.log<f32>, !lo_spn.log<f32>) -> !lo_spn.log<f32>
// CHECK:             %[[VAL_200:.*]] = "lo_spn.mul"(%[[VAL_199]], %[[VAL_194]]) : (!lo_spn.log<f32>, !lo_spn.log<f32>) -> !lo_spn.log<f32>
// CHECK:             %[[VAL_201:.*]] = "lo_spn.constant"() {type = !lo_spn.log<f32>, value = 1.000000e-01 : f64} : () -> !lo_spn.log<f32>
// CHECK:             %[[VAL_202:.*]] = "lo_spn.mul"(%[[VAL_200]], %[[VAL_201]]) : (!lo_spn.log<f32>, !lo_spn.log<f32>) -> !lo_spn.log<f32>
// CHECK:             %[[VAL_203:.*]] = "lo_spn.add"(%[[VAL_198]], %[[VAL_202]]) : (!lo_spn.log<f32>, !lo_spn.log<f32>) -> !lo_spn.log<f32>
// CHECK:             %[[VAL_204:.*]] = "lo_spn.strip_log"(%[[VAL_203]]) {target = f32} : (!lo_spn.log<f32>) -> f32
// CHECK:             "lo_spn.batch_write"(%[[VAL_204]], %[[VAL_1]], %[[VAL_182]]) : (f32, memref<?xf32>, index) -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func @spn_vector(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<?x6xf32>,
// CHECK-SAME:                     %[[VAL_1:.*]]: memref<?xf32>) {
// CHECK:           %[[VAL_2:.*]] = constant 0 : index
// CHECK:           %[[VAL_3:.*]] = memref.dim %[[VAL_0]], %[[VAL_2]] : memref<?x6xf32>
// CHECK:           %[[VAL_4:.*]] = memref.alloc(%[[VAL_3]]) : memref<?xf32>
// CHECK:           call @vec_task_0(%[[VAL_0]], %[[VAL_4]]) : (memref<?x6xf32>, memref<?xf32>) -> ()
// CHECK:           %[[VAL_5:.*]] = memref.tensor_load %[[VAL_4]] : memref<?xf32>
// CHECK:           %[[VAL_6:.*]] = memref.buffer_cast %[[VAL_5]] : memref<?xf32>
// CHECK:           "lo_spn.copy"(%[[VAL_6]], %[[VAL_1]]) : (memref<?xf32>, memref<?xf32>) -> ()
// CHECK:           "lo_spn.return"() : () -> ()
// CHECK:         }
