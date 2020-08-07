// RUN: %optcall --spn-simplify --spn-canonicalize --spn-lowering-to-standard %s | FileCheck %s

module {
  func @spn_kernel_body(%arg0: i32, %arg1: i32) -> f64 {
    %0 = "spn.histogram"(%arg0) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i64, ub = 1 : i64, val = 2.500000e-01 : f64}, {lb = 1 : i64, ub = 2 : i64, val = 7.500000e-01 : f64}]} : (i32) -> f64
    %1 = "spn.histogram"(%arg1) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i64, ub = 1 : i64, val = 2.500000e-01 : f64}, {lb = 1 : i64, ub = 2 : i64, val = 7.500000e-01 : f64}]} : (i32) -> f64
    %2 = "spn.histogram"(%arg0) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i64, ub = 1 : i64, val = 2.500000e-01 : f64}, {lb = 1 : i64, ub = 2 : i64, val = 7.500000e-01 : f64}]} : (i32) -> f64
    %3 = "spn.histogram"(%arg1) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i64, ub = 1 : i64, val = 2.500000e-01 : f64}, {lb = 1 : i64, ub = 2 : i64, val = 7.500000e-01 : f64}]} : (i32) -> f64
    %4 = "spn.histogram"(%arg0) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i64, ub = 1 : i64, val = 2.500000e-01 : f64}, {lb = 1 : i64, ub = 2 : i64, val = 7.500000e-01 : f64}]} : (i32) -> f64
    %5 = "spn.product"(%0, %1, %2, %3, %4) {opCount = 5 : ui32} : (f64, f64, f64, f64, f64) -> f64
    "spn.return"(%5) : (f64) -> ()
  }
}

//  N-ary products will be converted into a sequence of N-1 "mulf" insn.
//  This test uses histograms as "non-const" f64 values. We make sure that each
//  value / intermediate product is used and the last product is returned.

// CHECK-LABEL: @spn_kernel_body
// CHECK-SAME: [[ARGUMENT_0:%[^( ,:!]+]]
// CHECK-SAME: [[ARGUMENT_1:%[^( ,:!]+]]

// CHECK-DAG: [[HISTOGRAM_0:%[0-9]+]] = "spn.histvalue"()
// CHECK-DAG: [[HISTOGRAM_1:%[0-9]+]] = "spn.histvalue"()
// CHECK-DAG: [[HISTOGRAM_2:%[0-9]+]] = "spn.histvalue"()
// CHECK-DAG: [[HISTOGRAM_3:%[0-9]+]] = "spn.histvalue"()
// CHECK-DAG: [[HISTOGRAM_4:%[0-9]+]] = "spn.histvalue"()

// CHECK-DAG: [[INDEX_0:%[0-9]+]] = index_cast [[ARGUMENT_0]]
// CHECK-DAG: [[INDEX_1:%[0-9]+]] = index_cast [[ARGUMENT_1]]
// CHECK-DAG: [[INDEX_2:%[0-9]+]] = index_cast [[ARGUMENT_0]]
// CHECK-DAG: [[INDEX_3:%[0-9]+]] = index_cast [[ARGUMENT_1]]
// CHECK-DAG: [[INDEX_4:%[0-9]+]] = index_cast [[ARGUMENT_0]]

// CHECK-DAG: [[VALUE_0:%[0-9]+]] = load [[HISTOGRAM_0]]{{(\[{1})}}[[INDEX_0]]{{(\]{1})}} : memref
// CHECK-DAG: [[VALUE_1:%[0-9]+]] = load [[HISTOGRAM_1]]{{(\[{1})}}[[INDEX_1]]{{(\]{1})}} : memref
// CHECK-DAG: [[VALUE_2:%[0-9]+]] = load [[HISTOGRAM_2]]{{(\[{1})}}[[INDEX_2]]{{(\]{1})}} : memref
// CHECK-DAG: [[VALUE_3:%[0-9]+]] = load [[HISTOGRAM_3]]{{(\[{1})}}[[INDEX_3]]{{(\]{1})}} : memref
// CHECK-DAG: [[VALUE_4:%[0-9]+]] = load [[HISTOGRAM_4]]{{(\[{1})}}[[INDEX_4]]{{(\]{1})}} : memref

// CHECK-DAG: [[PRODUCT_0:%[0-9]+]] = mulf
// CHECK-DAG: [[PRODUCT_1:%[0-9]+]] = mulf
// CHECK-DAG: [[PRODUCT_2:%[0-9]+]] = mulf
// CHECK-DAG: [[PRODUCT_FINAL:%[0-9]+]] = mulf
// CHECK-DAG: [[VALUE_0]]
// CHECK-DAG: [[VALUE_1]]
// CHECK-DAG: [[VALUE_2]]
// CHECK-DAG: [[VALUE_3]]
// CHECK-DAG: [[VALUE_4]]
// CHECK-DAG: [[PRODUCT_0]]
// CHECK-DAG: [[PRODUCT_1]]
// CHECK-DAG: [[PRODUCT_2]]

// CHECK-NOT: mulf
// CHECK-NEXT: return [[PRODUCT_FINAL]]
