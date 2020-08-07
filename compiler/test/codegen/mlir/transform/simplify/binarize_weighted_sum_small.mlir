// RUN: %optcall --spn-simplify %s | FileCheck %s

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
    %8 = "spn.histogram"(%0) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i64, ub = 1 : i64, val = 2.500000e-01 : f64}, {lb = 1 : i64, ub = 2 : i64, val = 7.500000e-01 : f64}]} : (i32) -> f64
    %9 = "spn.histogram"(%1) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i64, ub = 1 : i64, val = 2.500000e-01 : f64}, {lb = 1 : i64, ub = 2 : i64, val = 7.500000e-01 : f64}]} : (i32) -> f64
    %10 = "spn.product"(%8, %9) {opCount = 2 : ui32} : (f64, f64) -> f64
    %11 = "spn.histogram"(%0) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i64, ub = 1 : i64, val = 2.500000e-01 : f64}, {lb = 1 : i64, ub = 2 : i64, val = 7.500000e-01 : f64}]} : (i32) -> f64
    %12 = "spn.histogram"(%1) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i64, ub = 1 : i64, val = 2.500000e-01 : f64}, {lb = 1 : i64, ub = 2 : i64, val = 7.500000e-01 : f64}]} : (i32) -> f64
    %13 = "spn.product"(%11, %12) {opCount = 2 : ui32} : (f64, f64) -> f64
    %14 = "spn.histogram"(%0) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i64, ub = 1 : i64, val = 2.500000e-01 : f64}, {lb = 1 : i64, ub = 2 : i64, val = 7.500000e-01 : f64}]} : (i32) -> f64
    %15 = "spn.histogram"(%1) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i64, ub = 1 : i64, val = 2.500000e-01 : f64}, {lb = 1 : i64, ub = 2 : i64, val = 7.500000e-01 : f64}]} : (i32) -> f64
    %16 = "spn.product"(%14, %15) {opCount = 2 : ui32} : (f64, f64) -> f64
    %17 = "spn.weighted_sum"(%4, %7, %10, %13, %16) {opCount = 5 : ui32, weights = [2.000000e-01, 2.000000e-01, 2.000000e-01, 2.000000e-01, 2.000000e-01]} : (f64, f64, f64, f64, f64) -> f64
    "spn.return"(%17) : (f64) -> ()
  }
  func @spn_kernel(%arg0: tensor<2xi32>) -> f64 {
    %0 = "spn.query.single"(%arg0) {spn = @spn_kernel_body} : (tensor<2xi32>) -> f64
    "spn.return"(%0) : (f64) -> ()
  }
}

//  This small test checks if a weighted sum is binarized (and split).
//  <constant, product> (one for each weight and operand) and one ordinary sum.
//
// ==== ==== ==== BEFORE ==== ==== ====
//
//          %18
//          +
//          %20
//          +
//          %22
//  %31 <--	+
//          %24
//          +
//          %26
//
// ==== ==== ==== AFTER ==== ==== ====
//
//                          %18
//                  %27 <--	+
//                  |       %20
//          %29	<--	+
//          |       |
//          |       %28	<--	%22
//  %31 <--	+
//          |       %24
//          |       |
//          %30 <--	+
//                  |
//                  %26
//
// To allow some reorderings we use DAG checks and divide them by using NOT.
// E.g. LVL2_OP2_REG1 (string subst) corresponds to '%27' and LVL2_OP1_REG1 to
// '%28', but they may be defined in different orderings. Only after they have
// been defined they may be used, which is why we divide the DAG checks from
// each other.
// #LVL1_OP2_REG1 (numeric var) should then evaluate to 29. Either the same or
// following line should use the registers LVL2_OP2_REG1 and LVL2_OP1_REG1.
// Also %30 should be a spn.sum as well. After %29, %30 have been defined they
// have to be used when defining %31.

// CHECK-LABEL: @spn_kernel_body
// CHECK-COUNT-2: "spn.input_var"(%arg{{[0-9]+}})
// CHECK-COUNT-10: "spn.histogram"(%{{[0-9]+}})
// CHECK-NEXT: "spn.product"(%{{[0-9]+}}, %{{[0-9]+}}) {{[[:graph:][:blank:]]+}}
// CHECK-NOT: "spn.weighted_sum"
// CHECK-COUNT-5: {{[[:blank:]]%[0-9]+ = "spn.constant"\(\) [[:graph:][:blank:]]+([[:space:]]{1})[[:blank:]]*%[0-9]+ = "spn.product"\(%[0-9]+, %[0-9]+\) [[:graph:][:blank:]]+}}
// CHECK-NOT: "spn.weighted_sum"
// CHECK-DAG: [[LVL2_OP2_REG1:%[0-9]+]] = "spn.sum"({{%[0-9]+}}, {{%[0-9]+}}) {opCount = 2
// CHECK-DAG: [[LVL2_OP1_REG1:%[0-9]+]] = "spn.sum"({{%[0-9]+}}) {opCount = 1
// CHECK-NOT: NOT
// CHECK-DAG: %[[#LVL1_OP2_REG1:]] = "spn.sum"(
// CHECK-DAG: %[[#LVL1_OP2_REG1+1]] = "spn.sum"(
// CHECK-DAG: [[LVL2_OP2_REG1]]
// CHECK-DAG: [[LVL2_OP1_REG1]]
// CHECK-SAME: ) {opCount = 2
// CHECK-NOT: NOT
// CHECK-NEXT: %[[#LVL1_OP2_REG1+2]] = "spn.sum"(
// CHECK-DAG: %[[#LVL1_OP2_REG1]]
// CHECK-DAG: %[[#LVL1_OP2_REG1+1]]
// CHECK-SAME: ) {opCount = 2
// CHECK-NEXT: "spn.return"
