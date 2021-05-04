// RUN: %optcall --vectorize-lospn-nodes %s | FileCheck %s

module  {
  func @vec_task_0(%arg0: memref<?x6xf64>, %arg1: memref<?xf64>) {
    %c0 = constant 0 : index
    %0 = dim %arg0, %c0 : memref<?x6xf64>
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
    %0 = dim %arg0, %c0 : memref<?x6xf64>
    %1 = alloc(%0) : memref<?xf64>
    call @vec_task_0(%arg0, %1) : (memref<?x6xf64>, memref<?xf64>) -> ()
    %2 = tensor_load %1 : memref<?xf64>
    %3 = tensor_to_memref %2 : memref<?xf64>
    "lo_spn.copy"(%3, %arg1) : (memref<?xf64>, memref<?xf64>) -> ()
    "lo_spn.return"() : () -> ()
  }
}

// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

// CHECK-LABEL:   global_memref "private" constant @histogram_vec_1 : memref<2xf64> = dense<[4.500000e-01, 5.500000e-01]>
// CHECK:         global_memref "private" constant @histogram_vec_0 : memref<2xf64> = dense<[2.500000e-01, 7.500000e-01]>
// CHECK:         global_memref "private" constant @categorical_vec_1 : memref<3xf64> = dense<[2.500000e-01, 6.250000e-01, 1.250000e-01]>
// CHECK:         global_memref "private" constant @categorical_vec_0 : memref<3xf64> = dense<[3.500000e-01, 5.500000e-01, 1.000000e-01]>

// CHECK-LABEL:   func @vec_task_0(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<?x6xf64>,
// CHECK-SAME:                     %[[VAL_1:.*]]: memref<?xf64>) {
// CHECK:           %[[VAL_2:.*]] = constant 0 : index
// CHECK:           %[[VAL_3:.*]] = dim %[[VAL_0]], %[[VAL_2]] : memref<?x6xf64>
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
// CHECK:             %[[VAL_19:.*]] = dim %[[VAL_0]], %[[VAL_18]] : memref<?x6xf64>
// CHECK:             %[[VAL_20:.*]] = constant 6 : index
// CHECK:             %[[VAL_21:.*]] = muli %[[VAL_19]], %[[VAL_20]] : index
// CHECK:             %[[VAL_22:.*]] = memref_reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_21]]], strides: [1] : memref<?x6xf64> to memref<?xf64>
// CHECK:             %[[VAL_23:.*]] = vector.gather %[[VAL_22]]{{\[}}%[[VAL_15]]], %[[VAL_17]], %[[VAL_16]] : memref<?xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
// CHECK:             %[[VAL_24:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_25:.*]] = vector.broadcast %[[VAL_24]] : i64 to vector<4xi64>
// CHECK:             %[[VAL_26:.*]] = constant dense<[1, 7, 13, 19]> : vector<4xi64>
// CHECK:             %[[VAL_27:.*]] = constant dense<6> : vector<4xi64>
// CHECK:             %[[VAL_28:.*]] = muli %[[VAL_25]], %[[VAL_27]] : vector<4xi64>
// CHECK:             %[[VAL_29:.*]] = addi %[[VAL_28]], %[[VAL_26]] : vector<4xi64>
// CHECK:             %[[VAL_30:.*]] = constant dense<0.000000e+00> : vector<4xf64>
// CHECK:             %[[VAL_31:.*]] = constant dense<true> : vector<4xi1>
// CHECK:             %[[VAL_32:.*]] = constant 0 : index
// CHECK:             %[[VAL_33:.*]] = dim %[[VAL_0]], %[[VAL_32]] : memref<?x6xf64>
// CHECK:             %[[VAL_34:.*]] = constant 6 : index
// CHECK:             %[[VAL_35:.*]] = muli %[[VAL_33]], %[[VAL_34]] : index
// CHECK:             %[[VAL_36:.*]] = memref_reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_35]]], strides: [1] : memref<?x6xf64> to memref<?xf64>
// CHECK:             %[[VAL_37:.*]] = vector.gather %[[VAL_36]]{{\[}}%[[VAL_29]]], %[[VAL_31]], %[[VAL_30]] : memref<?xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
// CHECK:             %[[VAL_38:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_39:.*]] = vector.broadcast %[[VAL_38]] : i64 to vector<4xi64>
// CHECK:             %[[VAL_40:.*]] = constant dense<[2, 8, 14, 20]> : vector<4xi64>
// CHECK:             %[[VAL_41:.*]] = constant dense<6> : vector<4xi64>
// CHECK:             %[[VAL_42:.*]] = muli %[[VAL_39]], %[[VAL_41]] : vector<4xi64>
// CHECK:             %[[VAL_43:.*]] = addi %[[VAL_42]], %[[VAL_40]] : vector<4xi64>
// CHECK:             %[[VAL_44:.*]] = constant dense<0.000000e+00> : vector<4xf64>
// CHECK:             %[[VAL_45:.*]] = constant dense<true> : vector<4xi1>
// CHECK:             %[[VAL_46:.*]] = constant 0 : index
// CHECK:             %[[VAL_47:.*]] = dim %[[VAL_0]], %[[VAL_46]] : memref<?x6xf64>
// CHECK:             %[[VAL_48:.*]] = constant 6 : index
// CHECK:             %[[VAL_49:.*]] = muli %[[VAL_47]], %[[VAL_48]] : index
// CHECK:             %[[VAL_50:.*]] = memref_reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_49]]], strides: [1] : memref<?x6xf64> to memref<?xf64>
// CHECK:             %[[VAL_51:.*]] = vector.gather %[[VAL_50]]{{\[}}%[[VAL_43]]], %[[VAL_45]], %[[VAL_44]] : memref<?xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
// CHECK:             %[[VAL_52:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_53:.*]] = vector.broadcast %[[VAL_52]] : i64 to vector<4xi64>
// CHECK:             %[[VAL_54:.*]] = constant dense<[3, 9, 15, 21]> : vector<4xi64>
// CHECK:             %[[VAL_55:.*]] = constant dense<6> : vector<4xi64>
// CHECK:             %[[VAL_56:.*]] = muli %[[VAL_53]], %[[VAL_55]] : vector<4xi64>
// CHECK:             %[[VAL_57:.*]] = addi %[[VAL_56]], %[[VAL_54]] : vector<4xi64>
// CHECK:             %[[VAL_58:.*]] = constant dense<0.000000e+00> : vector<4xf64>
// CHECK:             %[[VAL_59:.*]] = constant dense<true> : vector<4xi1>
// CHECK:             %[[VAL_60:.*]] = constant 0 : index
// CHECK:             %[[VAL_61:.*]] = dim %[[VAL_0]], %[[VAL_60]] : memref<?x6xf64>
// CHECK:             %[[VAL_62:.*]] = constant 6 : index
// CHECK:             %[[VAL_63:.*]] = muli %[[VAL_61]], %[[VAL_62]] : index
// CHECK:             %[[VAL_64:.*]] = memref_reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_63]]], strides: [1] : memref<?x6xf64> to memref<?xf64>
// CHECK:             %[[VAL_65:.*]] = vector.gather %[[VAL_64]]{{\[}}%[[VAL_57]]], %[[VAL_59]], %[[VAL_58]] : memref<?xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
// CHECK:             %[[VAL_66:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_67:.*]] = vector.broadcast %[[VAL_66]] : i64 to vector<4xi64>
// CHECK:             %[[VAL_68:.*]] = constant dense<[4, 10, 16, 22]> : vector<4xi64>
// CHECK:             %[[VAL_69:.*]] = constant dense<6> : vector<4xi64>
// CHECK:             %[[VAL_70:.*]] = muli %[[VAL_67]], %[[VAL_69]] : vector<4xi64>
// CHECK:             %[[VAL_71:.*]] = addi %[[VAL_70]], %[[VAL_68]] : vector<4xi64>
// CHECK:             %[[VAL_72:.*]] = constant dense<0.000000e+00> : vector<4xf64>
// CHECK:             %[[VAL_73:.*]] = constant dense<true> : vector<4xi1>
// CHECK:             %[[VAL_74:.*]] = constant 0 : index
// CHECK:             %[[VAL_75:.*]] = dim %[[VAL_0]], %[[VAL_74]] : memref<?x6xf64>
// CHECK:             %[[VAL_76:.*]] = constant 6 : index
// CHECK:             %[[VAL_77:.*]] = muli %[[VAL_75]], %[[VAL_76]] : index
// CHECK:             %[[VAL_78:.*]] = memref_reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_77]]], strides: [1] : memref<?x6xf64> to memref<?xf64>
// CHECK:             %[[VAL_79:.*]] = vector.gather %[[VAL_78]]{{\[}}%[[VAL_71]]], %[[VAL_73]], %[[VAL_72]] : memref<?xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
// CHECK:             %[[VAL_80:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_81:.*]] = vector.broadcast %[[VAL_80]] : i64 to vector<4xi64>
// CHECK:             %[[VAL_82:.*]] = constant dense<[5, 11, 17, 23]> : vector<4xi64>
// CHECK:             %[[VAL_83:.*]] = constant dense<6> : vector<4xi64>
// CHECK:             %[[VAL_84:.*]] = muli %[[VAL_81]], %[[VAL_83]] : vector<4xi64>
// CHECK:             %[[VAL_85:.*]] = addi %[[VAL_84]], %[[VAL_82]] : vector<4xi64>
// CHECK:             %[[VAL_86:.*]] = constant dense<0.000000e+00> : vector<4xf64>
// CHECK:             %[[VAL_87:.*]] = constant dense<true> : vector<4xi1>
// CHECK:             %[[VAL_88:.*]] = constant 0 : index
// CHECK:             %[[VAL_89:.*]] = dim %[[VAL_0]], %[[VAL_88]] : memref<?x6xf64>
// CHECK:             %[[VAL_90:.*]] = constant 6 : index
// CHECK:             %[[VAL_91:.*]] = muli %[[VAL_89]], %[[VAL_90]] : index
// CHECK:             %[[VAL_92:.*]] = memref_reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_91]]], strides: [1] : memref<?x6xf64> to memref<?xf64>
// CHECK:             %[[VAL_93:.*]] = vector.gather %[[VAL_92]]{{\[}}%[[VAL_85]]], %[[VAL_87]], %[[VAL_86]] : memref<?xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
// CHECK:             %[[VAL_94:.*]] = get_global_memref @categorical_vec_0 : memref<3xf64>
// CHECK:             %[[VAL_95:.*]] = fptoui %[[VAL_23]] : vector<4xf64> to vector<4xi64>
// CHECK:             %[[VAL_96:.*]] = constant dense<0.000000e+00> : vector<4xf64>
// CHECK:             %[[VAL_97:.*]] = constant dense<true> : vector<4xi1>
// CHECK:             %[[VAL_98:.*]] = vector.gather %[[VAL_94]]{{\[}}%[[VAL_95]]], %[[VAL_97]], %[[VAL_96]] : memref<3xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
// CHECK:             %[[VAL_99:.*]] = get_global_memref @categorical_vec_1 : memref<3xf64>
// CHECK:             %[[VAL_100:.*]] = fptoui %[[VAL_37]] : vector<4xf64> to vector<4xi64>
// CHECK:             %[[VAL_101:.*]] = constant dense<0.000000e+00> : vector<4xf64>
// CHECK:             %[[VAL_102:.*]] = constant dense<true> : vector<4xi1>
// CHECK:             %[[VAL_103:.*]] = vector.gather %[[VAL_99]]{{\[}}%[[VAL_100]]], %[[VAL_102]], %[[VAL_101]] : memref<3xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
// CHECK:             %[[VAL_104:.*]] = get_global_memref @histogram_vec_0 : memref<2xf64>
// CHECK:             %[[VAL_105:.*]] = fptoui %[[VAL_51]] : vector<4xf64> to vector<4xi64>
// CHECK:             %[[VAL_106:.*]] = constant dense<0.000000e+00> : vector<4xf64>
// CHECK:             %[[VAL_107:.*]] = constant dense<true> : vector<4xi1>
// CHECK:             %[[VAL_108:.*]] = vector.gather %[[VAL_104]]{{\[}}%[[VAL_105]]], %[[VAL_107]], %[[VAL_106]] : memref<2xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
// CHECK:             %[[VAL_109:.*]] = get_global_memref @histogram_vec_1 : memref<2xf64>
// CHECK:             %[[VAL_110:.*]] = fptoui %[[VAL_65]] : vector<4xf64> to vector<4xi64>
// CHECK:             %[[VAL_111:.*]] = constant dense<0.000000e+00> : vector<4xf64>
// CHECK:             %[[VAL_112:.*]] = constant dense<true> : vector<4xi1>
// CHECK:             %[[VAL_113:.*]] = vector.gather %[[VAL_109]]{{\[}}%[[VAL_110]]], %[[VAL_112]], %[[VAL_111]] : memref<2xf64>, vector<4xi64>, vector<4xi1>, vector<4xf64> into vector<4xf64>
// CHECK:             %[[VAL_114:.*]] = constant dense<0.3989422804014327> : vector<4xf64>
// CHECK:             %[[VAL_115:.*]] = constant dense<-5.000000e-01> : vector<4xf64>
// CHECK:             %[[VAL_116:.*]] = constant dense<5.000000e-01> : vector<4xf64>
// CHECK:             %[[VAL_117:.*]] = subf %[[VAL_79]], %[[VAL_116]] : vector<4xf64>
// CHECK:             %[[VAL_118:.*]] = mulf %[[VAL_117]], %[[VAL_117]] : vector<4xf64>
// CHECK:             %[[VAL_119:.*]] = mulf %[[VAL_118]], %[[VAL_115]] : vector<4xf64>
// CHECK:             %[[VAL_120:.*]] = math.exp %[[VAL_119]] : vector<4xf64>
// CHECK:             %[[VAL_121:.*]] = mulf %[[VAL_114]], %[[VAL_120]] : vector<4xf64>
// CHECK:             %[[VAL_122:.*]] = constant dense<3.9894228040143269> : vector<4xf64>
// CHECK:             %[[VAL_123:.*]] = constant dense<-49.999999999999993> : vector<4xf64>
// CHECK:             %[[VAL_124:.*]] = constant dense<2.500000e-01> : vector<4xf64>
// CHECK:             %[[VAL_125:.*]] = subf %[[VAL_93]], %[[VAL_124]] : vector<4xf64>
// CHECK:             %[[VAL_126:.*]] = mulf %[[VAL_125]], %[[VAL_125]] : vector<4xf64>
// CHECK:             %[[VAL_127:.*]] = mulf %[[VAL_126]], %[[VAL_123]] : vector<4xf64>
// CHECK:             %[[VAL_128:.*]] = math.exp %[[VAL_127]] : vector<4xf64>
// CHECK:             %[[VAL_129:.*]] = mulf %[[VAL_122]], %[[VAL_128]] : vector<4xf64>
// CHECK:             %[[VAL_130:.*]] = mulf %[[VAL_98]], %[[VAL_103]] : vector<4xf64>
// CHECK:             %[[VAL_131:.*]] = mulf %[[VAL_130]], %[[VAL_108]] : vector<4xf64>
// CHECK:             %[[VAL_132:.*]] = constant dense<1.000000e-01> : vector<4xf64>
// CHECK:             %[[VAL_133:.*]] = mulf %[[VAL_131]], %[[VAL_132]] : vector<4xf64>
// CHECK:             %[[VAL_134:.*]] = mulf %[[VAL_113]], %[[VAL_121]] : vector<4xf64>
// CHECK:             %[[VAL_135:.*]] = mulf %[[VAL_134]], %[[VAL_129]] : vector<4xf64>
// CHECK:             %[[VAL_136:.*]] = constant dense<1.000000e-01> : vector<4xf64>
// CHECK:             %[[VAL_137:.*]] = mulf %[[VAL_135]], %[[VAL_136]] : vector<4xf64>
// CHECK:             %[[VAL_138:.*]] = addf %[[VAL_133]], %[[VAL_137]] : vector<4xf64>
// CHECK:             %[[VAL_139:.*]] = math.log %[[VAL_138]] : vector<4xf64>
// CHECK:             vector.transfer_write %[[VAL_139]], %[[VAL_1]]{{\[}}%[[VAL_9]]] : vector<4xf64>, memref<?xf64>
// CHECK:           }
// CHECK:           %[[VAL_140:.*]] = constant 1 : index
// CHECK:           scf.for %[[VAL_141:.*]] = %[[VAL_6]] to %[[VAL_3]] step %[[VAL_140]] {
// CHECK:             %[[VAL_142:.*]] = "lo_spn.batch_read"(%[[VAL_0]], %[[VAL_141]]) {sampleIndex = 0 : ui32} : (memref<?x6xf64>, index) -> f64
// CHECK:             %[[VAL_143:.*]] = "lo_spn.batch_read"(%[[VAL_0]], %[[VAL_141]]) {sampleIndex = 1 : ui32} : (memref<?x6xf64>, index) -> f64
// CHECK:             %[[VAL_144:.*]] = "lo_spn.batch_read"(%[[VAL_0]], %[[VAL_141]]) {sampleIndex = 2 : ui32} : (memref<?x6xf64>, index) -> f64
// CHECK:             %[[VAL_145:.*]] = "lo_spn.batch_read"(%[[VAL_0]], %[[VAL_141]]) {sampleIndex = 3 : ui32} : (memref<?x6xf64>, index) -> f64
// CHECK:             %[[VAL_146:.*]] = "lo_spn.batch_read"(%[[VAL_0]], %[[VAL_141]]) {sampleIndex = 4 : ui32} : (memref<?x6xf64>, index) -> f64
// CHECK:             %[[VAL_147:.*]] = "lo_spn.batch_read"(%[[VAL_0]], %[[VAL_141]]) {sampleIndex = 5 : ui32} : (memref<?x6xf64>, index) -> f64
// CHECK:             %[[VAL_148:.*]] = "lo_spn.categorical"(%[[VAL_142]]) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false} : (f64) -> f64
// CHECK:             %[[VAL_149:.*]] = "lo_spn.categorical"(%[[VAL_143]]) {probabilities = [2.500000e-01, 6.250000e-01, 1.250000e-01], supportMarginal = false} : (f64) -> f64
// CHECK:             %[[VAL_150:.*]] = "lo_spn.histogram"(%[[VAL_144]]) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}], supportMarginal = false} : (f64) -> f64
// CHECK:             %[[VAL_151:.*]] = "lo_spn.histogram"(%[[VAL_145]]) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 4.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 5.500000e-01 : f64}], supportMarginal = false} : (f64) -> f64
// CHECK:             %[[VAL_152:.*]] = "lo_spn.gaussian"(%[[VAL_146]]) {mean = 5.000000e-01 : f64, stddev = 1.000000e+00 : f64, supportMarginal = false} : (f64) -> f64
// CHECK:             %[[VAL_153:.*]] = "lo_spn.gaussian"(%[[VAL_147]]) {mean = 2.500000e-01 : f64, stddev = 1.000000e-01 : f64, supportMarginal = false} : (f64) -> f64
// CHECK:             %[[VAL_154:.*]] = "lo_spn.mul"(%[[VAL_148]], %[[VAL_149]]) : (f64, f64) -> f64
// CHECK:             %[[VAL_155:.*]] = "lo_spn.mul"(%[[VAL_154]], %[[VAL_150]]) : (f64, f64) -> f64
// CHECK:             %[[VAL_156:.*]] = "lo_spn.constant"() {type = f64, value = 1.000000e-01 : f64} : () -> f64
// CHECK:             %[[VAL_157:.*]] = "lo_spn.mul"(%[[VAL_155]], %[[VAL_156]]) : (f64, f64) -> f64
// CHECK:             %[[VAL_158:.*]] = "lo_spn.mul"(%[[VAL_151]], %[[VAL_152]]) : (f64, f64) -> f64
// CHECK:             %[[VAL_159:.*]] = "lo_spn.mul"(%[[VAL_158]], %[[VAL_153]]) : (f64, f64) -> f64
// CHECK:             %[[VAL_160:.*]] = "lo_spn.constant"() {type = f64, value = 1.000000e-01 : f64} : () -> f64
// CHECK:             %[[VAL_161:.*]] = "lo_spn.mul"(%[[VAL_159]], %[[VAL_160]]) : (f64, f64) -> f64
// CHECK:             %[[VAL_162:.*]] = "lo_spn.add"(%[[VAL_157]], %[[VAL_161]]) : (f64, f64) -> f64
// CHECK:             %[[VAL_163:.*]] = "lo_spn.log"(%[[VAL_162]]) : (f64) -> f64
// CHECK:             "lo_spn.batch_write"(%[[VAL_163]], %[[VAL_1]], %[[VAL_141]]) : (f64, memref<?xf64>, index) -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func @spn_vector(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<?x6xf64>,
// CHECK-SAME:                     %[[VAL_1:.*]]: memref<?xf64>) {
// CHECK:           %[[VAL_2:.*]] = constant 0 : index
// CHECK:           %[[VAL_3:.*]] = dim %[[VAL_0]], %[[VAL_2]] : memref<?x6xf64>
// CHECK:           %[[VAL_4:.*]] = alloc(%[[VAL_3]]) : memref<?xf64>
// CHECK:           call @vec_task_0(%[[VAL_0]], %[[VAL_4]]) : (memref<?x6xf64>, memref<?xf64>) -> ()
// CHECK:           %[[VAL_5:.*]] = tensor_load %[[VAL_4]] : memref<?xf64>
// CHECK:           %[[VAL_6:.*]] = tensor_to_memref %[[VAL_5]] : memref<?xf64>
// CHECK:           "lo_spn.copy"(%[[VAL_6]], %[[VAL_1]]) : (memref<?xf64>, memref<?xf64>) -> ()
// CHECK:           "lo_spn.return"() : () -> ()
// CHECK:         }
