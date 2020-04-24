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
    %8 = "spn.weighted_sum"(%4, %7) {opCount = 2 : ui32, weights = [1.000000e+00, 1.000000e+00]} : (f64, f64) -> f64
    %9 = "spn.weighted_sum"(%4, %8) {opCount = 2 : ui32, weights = [7.500000e-01, 7.500000e-01]} : (f64, f64) -> f64
    "spn.return"(%9) : (f64) -> ()
  }
  func @spn_kernel(%arg0: tensor<2xi32>) -> f64 {
    %0 = "spn.query.single"(%arg0) {spn = @spn_kernel_body} : (tensor<2xi32>) -> f64
    "spn.return"(%0) : (f64) -> ()
  }
}

//  This small test checks if a weighted sum is rewritten into an ordinary sum,
//  if all weights are 1.0

// CHECK-LABEL: @spn_kernel_body
// CHECK-COUNT-2: "spn.input_var"(%arg{{[0-9]+}})
// CHECK-COUNT-4: "spn.histogram"(%{{[0-9]+}})
// CHECK-NEXT: "spn.product"(%{{[0-9]+}}, %{{[0-9]+}}) {{[[:graph:][:blank:]]+}}
// CHECK-NOT: "spn.weighted_sum"
// CHECK-NOT: "spn.constant"
// CHECK-NOT: "spn.product"
// CHECK-NEXT: "spn.sum"(%4, %7) {opCount = 2
// CHECK: "spn.return"
