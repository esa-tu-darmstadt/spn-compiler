// RUN: %optcall --spn-simplify --spn-canonicalize --spn-lowering-to-standard %s | FileCheck %s

module {
  func @spn_kernel_body(%arg0: i32, %arg1: i32) -> f64 {
    %0 = "spn.histogram"(%arg0) {bucketCount = 4 : ui32, buckets = [
      {lb = 0 : i64, ub = 1 : i64, val = 2.500000e-01 : f64},
      {lb = 1 : i64, ub = 2 : i64, val = 1.250000e-01 : f64},
      {lb = 2 : i64, ub = 3 : i64, val = 6.250000e-02 : f64},
      {lb = 3 : i64, ub = 4 : i64, val = 5.625000e-01 : f64}]} : (i32) -> f64
    "spn.return"(%0) : (f64) -> ()
  }
}

//  "spn.histogram" will be converted into "spn.histvalue" / memref.
//  The accessed index in the histogram becomes an index_cast.
//  In combination, it is possible to emit the corresponding load.

// CHECK-LABEL: @spn_kernel_body
// CHECK-DAG: [[HISTOGRAM:%[0-9]+]] = "spn.histvalue"()
// CHECK-DAG: [[INDEX:%[0-9]+]] = index_cast %arg0
// CHECK-NEXT: [[RET:%[0-9]+]] = load [[HISTOGRAM]]{{(\[{1})}}[[INDEX]]{{(\]{1})}} : memref
// CHECK-NEXT: return [[RET]]
