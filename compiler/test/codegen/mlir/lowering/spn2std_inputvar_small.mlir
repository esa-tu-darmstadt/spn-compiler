// RUN: %optcall --spn-simplify --spn-canonicalize --spn-lowering-to-standard %s | FileCheck %s

module {
  func @spn_kernel_body(%arg0: i32, %arg1: i32) -> f64 {
    %0 = "spn.input_var"(%arg0) {index = 0 : i32} : (i32) -> i32
    %1 = "spn.input_var"(%arg1) {index = 1 : i32} : (i32) -> i32
    %2 = addi %0, %1 : i32
    %3 = "spn.histogram"(%2) {bucketCount = 1 : ui32, buckets = [
      {lb = 0 : i64, ub = 1 : i64, val = 1.000000e-00 : f64}]} : (i32) -> f64
    "spn.return"(%3) : (f64) -> ()
  }
}

// Check if InputVars are directly replaced by their single argument. (see addi)

// CHECK-LABEL: @spn_kernel_body
// CHECK-SAME: [[ARGUMENT_0:%[^( ,:!]+]]
// CHECK-SAME: [[ARGUMENT_1:%[^( ,:!]+]]

// CHECK-NOT: spn
// CHECK-NEXT: [[INDEX_CALCULATED:%[0-9]+]] = addi
// CHECK-DAG: [[ARGUMENT_0]]
// CHECK-DAG: [[ARGUMENT_1]]
// CHECK: "spn.histvalue"()

// CHECK-NOT: spn
// CHECK: index_cast [[INDEX_CALCULATED]]

// CHECK-NOT: spn
// CHECK: return
