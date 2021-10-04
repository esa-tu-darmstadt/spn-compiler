// RUN: %optcall --vectorize-lospn-nodes %s | FileCheck %s

module  {
  func @vec_task_0(%arg0: memref<?x6xf64>, %arg1: memref<?xf64>) {
    %c0 = constant 0 : index
    %0 = memref.dim %arg0, %c0 : memref<?x6xf64>
    %c4 = constant 4 : index
    %1 = remi_unsigned %0, %c4 : index
    %2 = subi %0, %1 : index
    %c0_0 = constant 0 : index
    %c4_1 = constant 4 : index
    scf.for %arg2 = %c0_0 to %2 step %c4_1 {
      %3 = "lo_spn.batch_read"(%arg0, %arg2) {sampleIndex = 0 : ui32, vector_width = 4 : i32} : (memref<?x6xf64>, index) -> f64
      %4 = "lo_spn.batch_read"(%arg0, %arg2) {sampleIndex = 1 : ui32, vector_width = 4 : i32} : (memref<?x6xf64>, index) -> f64
      %5 = "lo_spn.batch_read"(%arg0, %arg2) {sampleIndex = 2 : ui32, vector_width = 4 : i32} : (memref<?x6xf64>, index) -> f64
      %6 = "lo_spn.batch_read"(%arg0, %arg2) {sampleIndex = 3 : ui32, vector_width = 4 : i32} : (memref<?x6xf64>, index) -> f64
      %7 = "lo_spn.batch_read"(%arg0, %arg2) {sampleIndex = 4 : ui32, vector_width = 4 : i32} : (memref<?x6xf64>, index) -> f64
      %8 = "lo_spn.batch_read"(%arg0, %arg2) {sampleIndex = 5 : ui32, vector_width = 4 : i32} : (memref<?x6xf64>, index) -> f64
      %9 = "lo_spn.categorical"(%3) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false, vector_width = 4 : i32} : (f64) -> f64
      %10 = "lo_spn.categorical"(%4) {probabilities = [2.500000e-01, 6.250000e-01, 1.250000e-01], supportMarginal = false, vector_width = 4 : i32} : (f64) -> f64
      %11 = "lo_spn.histogram"(%5) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}], supportMarginal = false, vector_width = 4 : i32} : (f64) -> f64
      %12 = "lo_spn.histogram"(%6) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 4.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 5.500000e-01 : f64}], supportMarginal = false, vector_width = 4 : i32} : (f64) -> f64
      %13 = "lo_spn.gaussian"(%7) {mean = 5.000000e-01 : f64, stddev = 1.000000e+00 : f64, supportMarginal = false, vector_width = 4 : i32} : (f64) -> f64
      %14 = "lo_spn.gaussian"(%8) {mean = 2.500000e-01 : f64, stddev = 1.000000e-01 : f64, supportMarginal = false, vector_width = 4 : i32} : (f64) -> f64
      %15 = "lo_spn.mul"(%9, %10) {vector_width = 4 : i32} : (f64, f64) -> f64
      %16 = "lo_spn.mul"(%15, %11) {vector_width = 4 : i32} : (f64, f64) -> f64
      %17 = "lo_spn.constant"() {type = f64, value = 1.000000e-01 : f64, vector_width = 4 : i32} : () -> f64
      %18 = "lo_spn.mul"(%16, %17) {vector_width = 4 : i32} : (f64, f64) -> f64
      %19 = "lo_spn.mul"(%12, %13) {vector_width = 4 : i32} : (f64, f64) -> f64
      %20 = "lo_spn.mul"(%19, %14) {vector_width = 4 : i32} : (f64, f64) -> f64
      %21 = "lo_spn.constant"() {type = f64, value = 1.000000e-01 : f64, vector_width = 4 : i32} : () -> f64
      %22 = "lo_spn.mul"(%20, %21) {vector_width = 4 : i32} : (f64, f64) -> f64
      %23 = "lo_spn.add"(%18, %22) {vector_width = 4 : i32} : (f64, f64) -> f64
      %24 = "lo_spn.log"(%23) {vector_width = 4 : i32} : (f64) -> f64
      "lo_spn.batch_write"(%24, %arg1, %arg2) {vector_width = 4 : i32} : (f64, memref<?xf64>, index) -> ()
    }
    %c1 = constant 1 : index
    scf.for %arg2 = %2 to %0 step %c1 {
      %3 = "lo_spn.batch_read"(%arg0, %arg2) {sampleIndex = 0 : ui32} : (memref<?x6xf64>, index) -> f64
      %4 = "lo_spn.batch_read"(%arg0, %arg2) {sampleIndex = 1 : ui32} : (memref<?x6xf64>, index) -> f64
      %5 = "lo_spn.batch_read"(%arg0, %arg2) {sampleIndex = 2 : ui32} : (memref<?x6xf64>, index) -> f64
      %6 = "lo_spn.batch_read"(%arg0, %arg2) {sampleIndex = 3 : ui32} : (memref<?x6xf64>, index) -> f64
      %7 = "lo_spn.batch_read"(%arg0, %arg2) {sampleIndex = 4 : ui32} : (memref<?x6xf64>, index) -> f64
      %8 = "lo_spn.batch_read"(%arg0, %arg2) {sampleIndex = 5 : ui32} : (memref<?x6xf64>, index) -> f64
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
      "lo_spn.batch_write"(%24, %arg1, %arg2) : (f64, memref<?xf64>, index) -> ()
    }
    return
  }
  func @spn_vector(%arg0: memref<?x6xf64>, %arg1: memref<?xf64>) {
    %c0 = constant 0 : index
    %0 = memref.dim %arg0, %c0 : memref<?x6xf64>
    %1 = memref.alloc(%0) : memref<?xf64>
    call @vec_task_0(%arg0, %1) : (memref<?x6xf64>, memref<?xf64>) -> ()
    %2 = memref.tensor_load %1 : memref<?xf64>
    %3 = memref.buffer_cast %2 : memref<?xf64>
    "lo_spn.copy"(%3, %arg1) : (memref<?xf64>, memref<?xf64>) -> ()
    "lo_spn.return"() : () -> ()
  }
}

// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

// CHECK-LABEL:   memref.global "private" constant @histogram_vec_1 : memref<2xf64> = dense<[4.500000e-01, 5.500000e-01]>
// CHECK:         memref.global "private" constant @histogram_vec_0 : memref<2xf64> = dense<[2.500000e-01, 7.500000e-01]>
// CHECK:         memref.global "private" constant @categorical_vec_1 : memref<3xf64> = dense<[2.500000e-01, 6.250000e-01, 1.250000e-01]>
// CHECK:         memref.global "private" constant @categorical_vec_0 : memref<3xf64> = dense<[3.500000e-01, 5.500000e-01, 1.000000e-01]>

// CHECK-LABEL:   func @vec_task_0(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<?x6xf64>,
// CHECK-SAME:                     %[[VAL_1:.*]]: memref<?xf64>) {
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
// CHECK:             %[[VAL_103:.*]] = constant dense<0> : vector<4xi64>
// CHECK:             %[[VAL_104:.*]] = constant dense<3> : vector<4xi64>
// CHECK:             %[[VAL_105:.*]] = cmpi sge, %[[VAL_101]], %[[VAL_103]] : vector<4xi64>
// CHECK:             %[[VAL_106:.*]] = cmpi slt, %[[VAL_101]], %[[VAL_104]] : vector<4xi64>
// CHECK:             %[[VAL_107:.*]] = and %[[VAL_105]], %[[VAL_106]] : vector<4xi1>
// CHECK:             %[[VAL_108:.*]] = constant dense<true> : vector<4xi1>
// CHECK:             %[[VAL_109:.*]] = constant dense<false> : vector<4xi1>
// CHECK:             %[[VAL_110:.*]] = select %[[VAL_107]], %[[VAL_108]], %[[VAL_109]] : vector<4xi1>, vector<4xi1>
// CHECK:             %[[VAL_111:.*]] = constant 0 : index
// CHECK:             %[[VAL_112:.*]] = vector.gather %[[VAL_100]]{{\[}}%[[VAL_111]]] {{\[}}%[[VAL_101]]], %[[VAL_110]], %[[VAL_102]] : memref<3xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
// CHECK:             %[[VAL_113:.*]] = memref.get_global @categorical_vec_1 : memref<3xf64>
// CHECK:             %[[VAL_114:.*]] = fptoui %[[VAL_39]] : vector<4xf64> to vector<4xi64>
// CHECK:             %[[VAL_115:.*]] = constant dense<0.000000e+00> : vector<4xf64>
// CHECK:             %[[VAL_116:.*]] = constant dense<0> : vector<4xi64>
// CHECK:             %[[VAL_117:.*]] = constant dense<3> : vector<4xi64>
// CHECK:             %[[VAL_118:.*]] = cmpi sge, %[[VAL_114]], %[[VAL_116]] : vector<4xi64>
// CHECK:             %[[VAL_119:.*]] = cmpi slt, %[[VAL_114]], %[[VAL_117]] : vector<4xi64>
// CHECK:             %[[VAL_120:.*]] = and %[[VAL_118]], %[[VAL_119]] : vector<4xi1>
// CHECK:             %[[VAL_121:.*]] = constant dense<true> : vector<4xi1>
// CHECK:             %[[VAL_122:.*]] = constant dense<false> : vector<4xi1>
// CHECK:             %[[VAL_123:.*]] = select %[[VAL_120]], %[[VAL_121]], %[[VAL_122]] : vector<4xi1>, vector<4xi1>
// CHECK:             %[[VAL_124:.*]] = constant 0 : index
// CHECK:             %[[VAL_125:.*]] = vector.gather %[[VAL_113]]{{\[}}%[[VAL_124]]] {{\[}}%[[VAL_114]]], %[[VAL_123]], %[[VAL_115]] : memref<3xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
// CHECK:             %[[VAL_126:.*]] = memref.get_global @histogram_vec_0 : memref<2xf64>
// CHECK:             %[[VAL_127:.*]] = fptoui %[[VAL_54]] : vector<4xf64> to vector<4xi64>
// CHECK:             %[[VAL_128:.*]] = constant dense<0.000000e+00> : vector<4xf64>
// CHECK:             %[[VAL_129:.*]] = constant dense<0> : vector<4xi64>
// CHECK:             %[[VAL_130:.*]] = constant dense<2> : vector<4xi64>
// CHECK:             %[[VAL_131:.*]] = cmpi sge, %[[VAL_127]], %[[VAL_129]] : vector<4xi64>
// CHECK:             %[[VAL_132:.*]] = cmpi slt, %[[VAL_127]], %[[VAL_130]] : vector<4xi64>
// CHECK:             %[[VAL_133:.*]] = and %[[VAL_131]], %[[VAL_132]] : vector<4xi1>
// CHECK:             %[[VAL_134:.*]] = constant dense<true> : vector<4xi1>
// CHECK:             %[[VAL_135:.*]] = constant dense<false> : vector<4xi1>
// CHECK:             %[[VAL_136:.*]] = select %[[VAL_133]], %[[VAL_134]], %[[VAL_135]] : vector<4xi1>, vector<4xi1>
// CHECK:             %[[VAL_137:.*]] = constant 0 : index
// CHECK:             %[[VAL_138:.*]] = vector.gather %[[VAL_126]]{{\[}}%[[VAL_137]]] {{\[}}%[[VAL_127]]], %[[VAL_136]], %[[VAL_128]] : memref<2xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
// CHECK:             %[[VAL_139:.*]] = memref.get_global @histogram_vec_1 : memref<2xf64>
// CHECK:             %[[VAL_140:.*]] = fptoui %[[VAL_69]] : vector<4xf64> to vector<4xi64>
// CHECK:             %[[VAL_141:.*]] = constant dense<0.000000e+00> : vector<4xf64>
// CHECK:             %[[VAL_142:.*]] = constant dense<0> : vector<4xi64>
// CHECK:             %[[VAL_143:.*]] = constant dense<2> : vector<4xi64>
// CHECK:             %[[VAL_144:.*]] = cmpi sge, %[[VAL_140]], %[[VAL_142]] : vector<4xi64>
// CHECK:             %[[VAL_145:.*]] = cmpi slt, %[[VAL_140]], %[[VAL_143]] : vector<4xi64>
// CHECK:             %[[VAL_146:.*]] = and %[[VAL_144]], %[[VAL_145]] : vector<4xi1>
// CHECK:             %[[VAL_147:.*]] = constant dense<true> : vector<4xi1>
// CHECK:             %[[VAL_148:.*]] = constant dense<false> : vector<4xi1>
// CHECK:             %[[VAL_149:.*]] = select %[[VAL_146]], %[[VAL_147]], %[[VAL_148]] : vector<4xi1>, vector<4xi1>
// CHECK:             %[[VAL_150:.*]] = constant 0 : index
// CHECK:             %[[VAL_151:.*]] = vector.gather %[[VAL_139]]{{\[}}%[[VAL_150]]] {{\[}}%[[VAL_140]]], %[[VAL_149]], %[[VAL_141]] : memref<2xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
// CHECK:             %[[VAL_152:.*]] = constant dense<0.3989422804014327> : vector<4xf64>
// CHECK:             %[[VAL_153:.*]] = constant dense<-5.000000e-01> : vector<4xf64>
// CHECK:             %[[VAL_154:.*]] = constant dense<5.000000e-01> : vector<4xf64>
// CHECK:             %[[VAL_155:.*]] = subf %[[VAL_84]], %[[VAL_154]] : vector<4xf64>
// CHECK:             %[[VAL_156:.*]] = mulf %[[VAL_155]], %[[VAL_155]] : vector<4xf64>
// CHECK:             %[[VAL_157:.*]] = mulf %[[VAL_156]], %[[VAL_153]] : vector<4xf64>
// CHECK:             %[[VAL_158:.*]] = math.exp %[[VAL_157]] : vector<4xf64>
// CHECK:             %[[VAL_159:.*]] = mulf %[[VAL_152]], %[[VAL_158]] : vector<4xf64>
// CHECK:             %[[VAL_160:.*]] = constant dense<3.9894228040143269> : vector<4xf64>
// CHECK:             %[[VAL_161:.*]] = constant dense<-49.999999999999993> : vector<4xf64>
// CHECK:             %[[VAL_162:.*]] = constant dense<2.500000e-01> : vector<4xf64>
// CHECK:             %[[VAL_163:.*]] = subf %[[VAL_99]], %[[VAL_162]] : vector<4xf64>
// CHECK:             %[[VAL_164:.*]] = mulf %[[VAL_163]], %[[VAL_163]] : vector<4xf64>
// CHECK:             %[[VAL_165:.*]] = mulf %[[VAL_164]], %[[VAL_161]] : vector<4xf64>
// CHECK:             %[[VAL_166:.*]] = math.exp %[[VAL_165]] : vector<4xf64>
// CHECK:             %[[VAL_167:.*]] = mulf %[[VAL_160]], %[[VAL_166]] : vector<4xf64>
// CHECK:             %[[VAL_168:.*]] = mulf %[[VAL_112]], %[[VAL_125]] : vector<4xf64>
// CHECK:             %[[VAL_169:.*]] = mulf %[[VAL_168]], %[[VAL_138]] : vector<4xf64>
// CHECK:             %[[VAL_170:.*]] = constant dense<1.000000e-01> : vector<4xf64>
// CHECK:             %[[VAL_171:.*]] = mulf %[[VAL_169]], %[[VAL_170]] : vector<4xf64>
// CHECK:             %[[VAL_172:.*]] = mulf %[[VAL_151]], %[[VAL_159]] : vector<4xf64>
// CHECK:             %[[VAL_173:.*]] = mulf %[[VAL_172]], %[[VAL_167]] : vector<4xf64>
// CHECK:             %[[VAL_174:.*]] = constant dense<1.000000e-01> : vector<4xf64>
// CHECK:             %[[VAL_175:.*]] = mulf %[[VAL_173]], %[[VAL_174]] : vector<4xf64>
// CHECK:             %[[VAL_176:.*]] = addf %[[VAL_171]], %[[VAL_175]] : vector<4xf64>
// CHECK:             %[[VAL_177:.*]] = math.log %[[VAL_176]] : vector<4xf64>
// CHECK:             vector.transfer_write %[[VAL_177]], %[[VAL_1]]{{\[}}%[[VAL_9]]] : vector<4xf64>, memref<?xf64>
// CHECK:           }
// CHECK:           %[[VAL_178:.*]] = constant 1 : index
// CHECK:           scf.for %[[VAL_179:.*]] = %[[VAL_6]] to %[[VAL_3]] step %[[VAL_178]] {
// CHECK:             %[[VAL_180:.*]] = "lo_spn.batch_read"(%[[VAL_0]], %[[VAL_179]]) {sampleIndex = 0 : ui32} : (memref<?x6xf64>, index) -> f64
// CHECK:             %[[VAL_181:.*]] = "lo_spn.batch_read"(%[[VAL_0]], %[[VAL_179]]) {sampleIndex = 1 : ui32} : (memref<?x6xf64>, index) -> f64
// CHECK:             %[[VAL_182:.*]] = "lo_spn.batch_read"(%[[VAL_0]], %[[VAL_179]]) {sampleIndex = 2 : ui32} : (memref<?x6xf64>, index) -> f64
// CHECK:             %[[VAL_183:.*]] = "lo_spn.batch_read"(%[[VAL_0]], %[[VAL_179]]) {sampleIndex = 3 : ui32} : (memref<?x6xf64>, index) -> f64
// CHECK:             %[[VAL_184:.*]] = "lo_spn.batch_read"(%[[VAL_0]], %[[VAL_179]]) {sampleIndex = 4 : ui32} : (memref<?x6xf64>, index) -> f64
// CHECK:             %[[VAL_185:.*]] = "lo_spn.batch_read"(%[[VAL_0]], %[[VAL_179]]) {sampleIndex = 5 : ui32} : (memref<?x6xf64>, index) -> f64
// CHECK:             %[[VAL_186:.*]] = "lo_spn.categorical"(%[[VAL_180]]) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false} : (f64) -> f64
// CHECK:             %[[VAL_187:.*]] = "lo_spn.categorical"(%[[VAL_181]]) {probabilities = [2.500000e-01, 6.250000e-01, 1.250000e-01], supportMarginal = false} : (f64) -> f64
// CHECK:             %[[VAL_188:.*]] = "lo_spn.histogram"(%[[VAL_182]]) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}], supportMarginal = false} : (f64) -> f64
// CHECK:             %[[VAL_189:.*]] = "lo_spn.histogram"(%[[VAL_183]]) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 4.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 5.500000e-01 : f64}], supportMarginal = false} : (f64) -> f64
// CHECK:             %[[VAL_190:.*]] = "lo_spn.gaussian"(%[[VAL_184]]) {mean = 5.000000e-01 : f64, stddev = 1.000000e+00 : f64, supportMarginal = false} : (f64) -> f64
// CHECK:             %[[VAL_191:.*]] = "lo_spn.gaussian"(%[[VAL_185]]) {mean = 2.500000e-01 : f64, stddev = 1.000000e-01 : f64, supportMarginal = false} : (f64) -> f64
// CHECK:             %[[VAL_192:.*]] = "lo_spn.mul"(%[[VAL_186]], %[[VAL_187]]) : (f64, f64) -> f64
// CHECK:             %[[VAL_193:.*]] = "lo_spn.mul"(%[[VAL_192]], %[[VAL_188]]) : (f64, f64) -> f64
// CHECK:             %[[VAL_194:.*]] = "lo_spn.constant"() {type = f64, value = 1.000000e-01 : f64} : () -> f64
// CHECK:             %[[VAL_195:.*]] = "lo_spn.mul"(%[[VAL_193]], %[[VAL_194]]) : (f64, f64) -> f64
// CHECK:             %[[VAL_196:.*]] = "lo_spn.mul"(%[[VAL_189]], %[[VAL_190]]) : (f64, f64) -> f64
// CHECK:             %[[VAL_197:.*]] = "lo_spn.mul"(%[[VAL_196]], %[[VAL_191]]) : (f64, f64) -> f64
// CHECK:             %[[VAL_198:.*]] = "lo_spn.constant"() {type = f64, value = 1.000000e-01 : f64} : () -> f64
// CHECK:             %[[VAL_199:.*]] = "lo_spn.mul"(%[[VAL_197]], %[[VAL_198]]) : (f64, f64) -> f64
// CHECK:             %[[VAL_200:.*]] = "lo_spn.add"(%[[VAL_195]], %[[VAL_199]]) : (f64, f64) -> f64
// CHECK:             %[[VAL_201:.*]] = "lo_spn.log"(%[[VAL_200]]) : (f64) -> f64
// CHECK:             "lo_spn.batch_write"(%[[VAL_201]], %[[VAL_1]], %[[VAL_179]]) : (f64, memref<?xf64>, index) -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func @spn_vector(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<?x6xf64>,
// CHECK-SAME:                     %[[VAL_1:.*]]: memref<?xf64>) {
// CHECK:           %[[VAL_2:.*]] = constant 0 : index
// CHECK:           %[[VAL_3:.*]] = memref.dim %[[VAL_0]], %[[VAL_2]] : memref<?x6xf64>
// CHECK:           %[[VAL_4:.*]] = memref.alloc(%[[VAL_3]]) : memref<?xf64>
// CHECK:           call @vec_task_0(%[[VAL_0]], %[[VAL_4]]) : (memref<?x6xf64>, memref<?xf64>) -> ()
// CHECK:           %[[VAL_5:.*]] = memref.tensor_load %[[VAL_4]] : memref<?xf64>
// CHECK:           %[[VAL_6:.*]] = memref.buffer_cast %[[VAL_5]] : memref<?xf64>
// CHECK:           "lo_spn.copy"(%[[VAL_6]], %[[VAL_1]]) : (memref<?xf64>, memref<?xf64>) -> ()
// CHECK:           "lo_spn.return"() : () -> ()
// CHECK:         }
