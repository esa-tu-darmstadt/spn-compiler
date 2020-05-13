// RUN: %optcall --spn-canonicalize %s | FileCheck %s

module {
  func @spn_kernel_body(%arg0: i32, %arg1: i32) -> f64 {
    %0 = "spn.constant"() {value = 5.00000e-01 : f64} : () -> f64
    %1 = "spn.constant"() {value = -5.00000e-01 : f64} : () -> f64
    %2 = "spn.sum"(%0) {opCount = 1 : ui32} : (f64) -> f64
    %3 = "spn.sum"(%2) {opCount = 1 : ui32} : (f64) -> f64
    %4 = "spn.sum"(%2, %3) {opCount = 2 : ui32} : (f64, f64) -> f64
    %5 = "spn.sum"(%4) {opCount = 1 : ui32} : (f64) -> f64
    %6 = "spn.sum"(%5, %1, %1) {opCount = 3 : ui32} : (f64, f64, f64) -> f64
    "spn.return"(%6) : (f64) -> ()
  }
  func @spn_kernel(%arg0: tensor<2xi32>) -> f64 {
    %0 = "spn.query.single"(%arg0) {spn = @spn_kernel_body} : (tensor<2xi32>) -> f64
    "spn.return"(%0) : (f64) -> ()
  }
}

//  This small test checks if a sum is reduced if there is only one operand.
//  In the example %6 will be constant (0.0) and can be replaced.
//  This is achieved by multiple folds & reductions, converging at const 0.0.

// CHECK-LABEL: @spn_kernel_body
// CHECK-NEXT: "spn.constant"
// CHECK-NEXT: "spn.return"
