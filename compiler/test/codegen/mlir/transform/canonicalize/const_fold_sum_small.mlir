// RUN: %optcall --spn-canonicalize %s | FileCheck %s

module {
  func @spn_kernel_body(%arg0: i32, %arg1: i32) -> f64 {
    %0 = "spn.constant"() {value = 5.00000e-01 : f64} : () -> f64
    %1 = "spn.constant"() {value = 2.50000e-01 : f64} : () -> f64
    %2 = "spn.constant"() {value = 1.25000e-01 : f64} : () -> f64
    %3 = "spn.constant"() {value = 6.25000e-02 : f64} : () -> f64
    %4 = "spn.sum"(%0, %1, %2) {opCount = 3 : ui32} : (f64, f64, f64) -> f64
    %5 = "spn.sum"(%3, %3) {opCount = 2 : ui32} : (f64, f64) -> f64
    %6 = "spn.sum"(%4, %5) {opCount = 2 : ui32} : (f64, f64) -> f64
    "spn.return"(%6) : (f64) -> ()
  }
  func @spn_kernel(%arg0: tensor<2xi32>) -> f64 {
    %0 = "spn.query.single"(%arg0) {spn = @spn_kernel_body} : (tensor<2xi32>) -> f64
    "spn.return"(%0) : (f64) -> ()
  }
}

//  This small test checks if a sum is fold if there are constant operands.
//  In the example %6 will be constant (1.0) and can be replaced.
//  This is achieved by multiple sum-op folds, converging at const 1.0.

// CHECK-LABEL: @spn_kernel_body
// CHECK-NEXT: "spn.constant"
// CHECK-NEXT: "spn.return"
