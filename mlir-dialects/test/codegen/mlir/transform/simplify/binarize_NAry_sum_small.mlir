// RUN: %optcall --spn-simplify %s | FileCheck %s

module {
  "spn.single_joint"() ( {
    ^bb0(%arg0: ui32, %arg1: ui32): // no predecessors
      %2 = "spn.histogram"(%arg0) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}]} : (ui32) -> !spn.probability
      %3 = "spn.histogram"(%arg1) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}]} : (ui32) -> !spn.probability
      %4 = "spn.product"(%2, %3) : (!spn.probability, !spn.probability) -> !spn.probability
      %5 = "spn.histogram"(%arg0) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}]} : (ui32) -> !spn.probability
      %6 = "spn.histogram"(%arg1) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}]} : (ui32) -> !spn.probability
      %7 = "spn.product"(%5, %6) : (!spn.probability, !spn.probability) -> !spn.probability
      %8 = "spn.histogram"(%arg0) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}]} : (ui32) -> !spn.probability
      %9 = "spn.histogram"(%arg1) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}]} : (ui32) -> !spn.probability
      %10 = "spn.product"(%8, %9) : (!spn.probability, !spn.probability) -> !spn.probability
      %11 = "spn.histogram"(%arg0) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}]} : (ui32) -> !spn.probability
      %12 = "spn.histogram"(%arg1) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}]} : (ui32) -> !spn.probability
      %13 = "spn.product"(%11, %12) : (!spn.probability, !spn.probability) -> !spn.probability
      %14 = "spn.histogram"(%arg0) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}]} : (ui32) -> !spn.probability
      %15 = "spn.histogram"(%arg1) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}]} : (ui32) -> !spn.probability
      %16 = "spn.product"(%14, %15) : (!spn.probability, !spn.probability) -> !spn.probability
      %17 = "spn.sum"(%4, %7, %10, %13, %16) : (!spn.probability, !spn.probability, !spn.probability, !spn.probability, !spn.probability) -> !spn.probability
      "spn.return"(%17) : (!spn.probability) -> ()
  }) {inputType = ui32, numFeatures = 2 : ui32} : () -> ()
}

//  This small test checks if a sum (with 5 operands) is binarized.
//
// ==== ==== ==== BEFORE ==== ==== ====
//
//          %16
//          +
//          %20
//          +
//          %22
//  %21 <--	+
//          %24
//          +
//          %26
//
// ==== ==== ==== AFTER ==== ==== ====
//
//                          %4
//                  %17 <--	+
//                  |       %7
//          %19	<--	+
//          |       |
//          |       %18	<--	%10
//  %21 <--	+
//          |       %13
//          |       |
//          %20 <--	+
//                  |
//                  %16
//
// To allow some reorderings we use DAG checks and divide them by using NOT.
// E.g. LVL2_OP2_REG1 (string subst) corresponds to '%17' and LVL2_OP1_REG1 to
// '%18', but they may be defined in different orderings. Only after they have
// been defined they may be used, which is why we divide the DAG checks from
// each other.
// #LVL1_OP2_REG1 (numeric var) should then evaluate to 19. Either the same or
// following line should use the registers LVL2_OP2_REG1 and LVL2_OP1_REG1.
// Also %30 should be a spn.sum as well. After %19, %20 have been defined
// they have to be used when defining %21.

// CHECK-LABEL: ^bb0
// CHECK-COUNT-10: "spn.histogram"(%arg{{[0-9]+}})
// CHECK-NEXT: "spn.product"(%{{[0-9]+}}, %{{[0-9]+}})
// CHECK-NOT: "spn.sum"({{%[0-9]+((, %[0-9]+){2,})}})
// CHECK-NOT: "spn.product"
// CHECK-DAG: [[LVL2_OP2_REG1:%[0-9]+]] = "spn.sum"({{%[0-9]+}}, {{%[0-9]+}})
// CHECK-DAG: [[LVL2_OP1_REG1:%[0-9]+]] = "spn.sum"({{%[0-9]+}})
// CHECK-NOT: NOT
// CHECK-DAG: %[[#LVL1_OP2_REG1:]] = "spn.sum"(
// CHECK-DAG: %[[#LVL1_OP2_REG1+1]] = "spn.sum"(
// CHECK-DAG: [[LVL2_OP2_REG1]]
// CHECK-DAG: [[LVL2_OP1_REG1]]
// CHECK-NOT: NOT
// CHECK-NEXT: %[[#LVL1_OP2_REG1+2]] = "spn.sum"(
// CHECK-DAG: %[[#LVL1_OP2_REG1]]
// CHECK-DAG: %[[#LVL1_OP2_REG1+1]]
// CHECK-NEXT: "spn.return"
