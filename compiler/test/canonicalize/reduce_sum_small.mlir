// RUN: %optcall --spn-canonicalize %s | FileCheck %s

module {
  func @spn_kernel_body(%arg0: i32, %arg1: i32) -> f64 {
    %0 = "spn.constant"() {value = 5.00000e-01 : f64} : () -> f64
    %1 = "spn.sum"(%0) {opCount = 1 : ui32} : (f64) -> f64
    %2 = "spn.sum"(%1) {opCount = 1 : ui32} : (f64) -> f64
    %3 = "spn.sum"(%1, %2) {opCount = 2 : ui32} : (f64, f64) -> f64
    %4 = "spn.sum"() {opCount = 0 : ui32} : () -> f64
    %5 = "spn.sum"(%3, %4) {opCount = 2 : ui32} : (f64, f64) -> f64
    "spn.return"(%3) : (f64) -> () //* SAFE *//
//    "spn.return"(%5) : (f64) -> () //* UN-SAFE *//
  }
  func @spn_kernel(%arg0: tensor<2xi32>) -> f64 {
    %0 = "spn.query.single"(%arg0) {spn = @spn_kernel_body} : (tensor<2xi32>) -> f64
    "spn.return"(%0) : (f64) -> ()
  }
}

//  This small test checks if a sum is reudced if there are constants.
//  Operations should also be reduced when only one or zero operands
//  are provided -- either simply return the parameter or delete the op.
//  In the example %5 will be constant (1.0) and can be replaced.
//  This is achieved by multiple sum-op reductions, converging at const 1.0.

// @TODO: ATTENTION: Currently "disabled" by the return of another (%3) value!
// -> Potential bug?!

// CHECK-LABEL: @spn_kernel_body
// CHECK-NEXT: "spn.constant"
// CHECK-NEXT: "spn.return"
