// RUN: %optcall --spn-canonicalize %s | FileCheck %s

module {
  func @spn_kernel_body(%arg0: i32, %arg1: i32) -> f64 {
    %0 = "spn.constant"() {value = 2.0 : f64} : () -> f64
    %1 = "spn.constant"() {value = 4.0 : f64} : () -> f64
    %2 = "spn.product"(%0, %1) {opCount = 2 : ui32} : (f64, f64) -> f64
    %3 = "spn.product"(%2) {opCount = 1 : ui32} : (f64) -> f64
    %4 = "spn.product"(%3) {opCount = 1 : ui32} : (f64) -> f64
    %5 = "spn.constant"() {value = 5.00000e-01 : f64} : () -> f64
    %6 = "spn.constant"() {value = 2.50000e-01 : f64} : () -> f64
    %7 = "spn.constant"() {value = 1.25000e-01 : f64} : () -> f64
    %8 = "spn.product"(%0, %6, %6, %2, %3, %4, %7, %7, %7) {opCount = 9 : ui32} : (f64, f64, f64, f64, f64, f64, f64, f64, f64) -> f64
    %9 = "spn.product"(%4, %8) {opCount = 2 : ui32} : (f64, f64) -> f64
    "spn.return"(%8) : (f64) -> ()  //* SAFE *//
//    "spn.return"(%9) : (f64) -> ()  //* UN-SAFE *//
  }
  func @spn_kernel(%arg0: tensor<2xi32>) -> f64 {
    %0 = "spn.query.single"(%arg0) {spn = @spn_kernel_body} : (tensor<2xi32>) -> f64
    "spn.return"(%0) : (f64) -> ()
  }
}

//  This small test checks if a product is reduced if there are constant
//  operands. Operations should also be reduced when only one or zero operands
//  are provided -- either simply return the parameter or delete the op.
//  In the example %9 will be constant (1.0) and can be replaced.
//  This is achieved by multiple product-op folds, converging at const 1.0.

// @TODO: ATTENTION: Currently "disabled" by the return of another (%8) value!
// -> Potential bug?!

// CHECK-LABEL: @spn_kernel_body
// CHECK-NEXT: "spn.constant"
// CHECK-NEXT: "spn.return"
