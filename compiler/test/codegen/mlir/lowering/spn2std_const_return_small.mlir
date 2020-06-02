// RUN: %optcall --spn-simplify --spn-canonicalize --spn-lowering-to-standard %s | FileCheck %s

module {
  func @spn_kernel_body(%arg0: i32, %arg1: i32) -> f64 {
    %0 = "spn.constant"() {value = 1.0 : f64} : () -> f64
    "spn.return"(%0) : (f64) -> ()
  }
}

//  Basic lowering of a SPN constant and SPN return operation.

// CHECK-LABEL: @spn_kernel_body
// CHECK-NOT: spn
// CHECK-NEXT: constant
// CHECK-SAME: 1.0
// CHECK-SAME: f64
// CHECK-NOT: spn
// CHECK-NEXT: return
