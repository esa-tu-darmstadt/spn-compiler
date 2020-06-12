// RUN: %optcall --spn-simplify --spn-canonicalize --spn-lowering-to-standard %s | FileCheck %s

module {
  func @spn_kernel_body(%arg0: i32, %arg1: i32) -> f64 {
    %0 = "spn.constant"() {value = 1.0 : f64} : () -> f64
    "spn.return"(%0) : (f64) -> ()
  }
  func @spn_kernel(%arg0: tensor<2xi32>) -> f64 {
    %0 = "spn.query.single"(%arg0) {spn = @spn_kernel_body} : (tensor<2xi32>) -> f64
    "spn.return"(%0) : (f64) -> ()
  }
}

//  Because of their relationship to the @spn_kernel func this test will cover
//  - FunctionLowering: tensors have to be lowered to memrefs
//  - SingleQueryLowering: queries become calls to the spn_kernel_body
//  - SPNOpLowering: 'tensor' (memref) elements have to be loaded and provided

// CHECK-LABEL: @spn_kernel_body
// CHECK-SAME: [[ARGUMENT_0:%[^( ,:!]+]]
// CHECK-SAME: [[ARGUMENT_1:%[^( ,:!]+]]
// CHECK-NOT: spn.
// CHECK: constant
// CHECK-NOT: spn.
// CHECK: return

// CHECK-LABEL: @spn_kernel

/// (*) FunctionLowering
// CHECK-SAME: [[ARGUMENT:%[^() ,:!<>x]+]]
// CHECK-SAME: [[TYPE:memref[^() ,:!]+]]
// CHECK-NOT: tensor

/// (*) SPNOpLowering
// CHECK-DAG: [[INDEX_0:%[a-zA-Z0-9]+]] = constant {{[0-9]+}}
// CHECK-DAG: [[INDEX_1:%[a-zA-Z0-9]+]] = constant {{[0-9]+}}
// CHECK-DAG: [[PARAM_0:%[0-9]+]] = load [[ARGUMENT]]{{(\[{1})}}[[INDEX_0]]{{(\]{1})}} : [[TYPE]]
// CHECK-DAG: [[PARAM_1:%[0-9]+]] = load [[ARGUMENT]]{{(\[{1})}}[[INDEX_1]]{{(\]{1})}} : [[TYPE]]

/// (*) SingleQueryLowering
// CHECK-NEXT: [[QUERY:%[0-9]+]] = call @spn_kernel_body
// CHECK-DAG: [[PARAM_0]]
// CHECK-DAG: [[PARAM_1]]

// CHECK-NOT: spn.
// CHECK-NEXT: return [[QUERY]]
