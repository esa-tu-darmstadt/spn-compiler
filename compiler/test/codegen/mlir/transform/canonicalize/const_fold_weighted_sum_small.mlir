// RUN: %optcall --spn-canonicalize %s | FileCheck %s

module {
  func @spn_kernel_body(%arg0: i32, %arg1: i32) -> f64 {
    %0 = "spn.input_var"(%arg0) {index = 0 : i32} : (i32) -> i32
    %1 = "spn.input_var"(%arg1) {index = 1 : i32} : (i32) -> i32
    %2 = "spn.histogram"(%0) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i64, ub = 1 : i64, val = 2.500000e-01 : f64}, {lb = 1 : i64, ub = 2 : i64, val = 7.500000e-01 : f64}]} : (i32) -> f64
    %3 = "spn.histogram"(%1) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i64, ub = 1 : i64, val = 2.500000e-01 : f64}, {lb = 1 : i64, ub = 2 : i64, val = 7.500000e-01 : f64}]} : (i32) -> f64
    %4 = "spn.product"(%2, %3) {opCount = 2 : ui32} : (f64, f64) -> f64
    %5 = "spn.histogram"(%0) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i64, ub = 1 : i64, val = 2.500000e-01 : f64}, {lb = 1 : i64, ub = 2 : i64, val = 7.500000e-01 : f64}]} : (i32) -> f64
    %6 = "spn.histogram"(%1) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i64, ub = 1 : i64, val = 2.500000e-01 : f64}, {lb = 1 : i64, ub = 2 : i64, val = 7.500000e-01 : f64}]} : (i32) -> f64
    %7 = "spn.product"(%5, %6) {opCount = 2 : ui32} : (f64, f64) -> f64
    %8 = "spn.weighted_sum"(%4, %7) {opCount = 2 : ui32, weights = [3.000000e-01, 0.69999999999999996]} : (f64, f64) -> f64
    %9 = "spn.constant"() {value = 7.0 : f64} : () -> f64
    %10 = "spn.constant"() {value = 23.0 : f64} : () -> f64
    %11 = "spn.constant"() {value = 42.0 : f64} : () -> f64
    %12 = "spn.product"(%9, %10) {opCount = 2 : ui32} : (f64, f64) -> f64
    %13 = "spn.product"(%10, %11) {opCount = 2 : ui32} : (f64, f64) -> f64
    %14 = "spn.weighted_sum"(%8, %12, %13) {opCount = 3 : ui32, weights = [1.250000e-01, 2.500000e-01, 5.000000e-01]} : (f64, f64, f64) -> f64
    "spn.return"(%14) : (f64) -> ()
  }
  func @spn_kernel(%arg0: tensor<2xi32>) -> f64 {
    %0 = "spn.query.single"(%arg0) {spn = @spn_kernel_body} : (tensor<2xi32>) -> f64
    "spn.return"(%0) : (f64) -> ()
  }
}

//  This small test checks if a weighted sum is fold if there are constant
//  operands. In the example %7 is dependent of external parameters (non-const).
//  But %9, %10, %11 are const, which is why %12, %13 are also const.
//  Therefore %14 can be fold into a 2-opCount weighted sum.
//  Note: 5.232500e+02 = ( 0.25 * [7 * 23]  +  0.5 * [23 * 42] )

// CHECK-LABEL: @spn_kernel_body
// CHECK-COUNT-2: "spn.input_var"(%arg{{[0-9]+}})
// CHECK-COUNT-4: "spn.histogram"(%{{[0-9]+}})
// CHECK-NEXT: "spn.product"(%{{[0-9]+}}, %{{[0-9]+}}) {{[[:graph:][:blank:]]+}}
// CHECK-NEXT: [[NON_CONST_ADDEND:%[0-9]+]] = "spn.weighted_sum"
// CHECK-NEXT: [[CONST_ADDEND:%[0-9]+]] = "spn.constant"()
// CHECK-SAME: value = 5.232500e+02 : f64
// CHECK-NOT: "spn.constant"
// CHECK-NOT: "spn.sum"
// CHECK-DAG: "spn.weighted_sum"(
// CHECK-DAG: [[NON_CONST_ADDEND]]
// CHECK-DAG: [[CONST_ADDEND]]
// CHECK-SAME: ) {opCount = 2
// CHECK-NEXT: "spn.return"
