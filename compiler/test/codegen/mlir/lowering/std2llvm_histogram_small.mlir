// RUN: %optcall --spn-simplify --spn-canonicalize --spn-lowering-to-standard --spn-lowering-to-llvm %s | FileCheck %s

module {
  func @spn_kernel_body(%arg0: i32, %arg1: i32) -> f64 {
    %0 = "spn.histogram"(%arg0) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i64, ub = 1 : i64, val = 2.500000e-01 : f64}, {lb = 1 : i64, ub = 2 : i64, val = 7.500000e-01 : f64}]} : (i32) -> f64
    "spn.return"(%0) : (f64) -> ()
  }
}

//  "spn.histvalue" will be converted into a sequence of native LLVM insn.
//  The accessed data will be held in a global variable. Its name (eg. @test_0)
//  ends at a 'space' or '(' -- look for every other char.
//  In the process, multiple indexes are calculated -- ultimately yielding a
//  pointer which is used in the final load.

// CHECK: module

// CHECK: llvm.mlir.global internal constant [[HISTOGRAM:@[^( ]+]]
// CHECK-SAME: dense
// CHECK-SAME: 2.500000e-01
// CHECK-SAME: 7.500000e-01
// CHECK-SAME: tensor<2xf64>

// CHECK-LABEL: @spn_kernel_body
// CHECK-SAME: ([[ARGUMENT_0:%[^( ,:!]+]]

// CHECK: [[ADDRESS:%[0-9]+]] = llvm.mlir.addressof [[HISTOGRAM]]
// CHECK: llvm.getelementptr [[ADDRESS]]
// CHECK: [[ARGUMENT_1:%[0-9]+]] = llvm.sext [[ARGUMENT_0]]
// CHECK: [[INDEX_0:%[0-9]+]] = llvm.mul
// CHECK: [[ARGUMENT_1]]
// CHECK: [[INDEX_1:%[0-9]+]] = llvm.add
// CHECK: [[INDEX_0]]
// CHECK: [[POINTER:%[0-9]+]] = llvm.getelementptr %{{[0-9]+}}{{(\[{1})}}[[INDEX_1]]{{(\]{1})}}
// CHECK-NEXT: [[VALUE:%[0-9]+]] = llvm.load [[POINTER]]
// CHECK-NEXT: llvm.return [[VALUE]]
