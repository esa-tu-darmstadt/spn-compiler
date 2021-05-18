// RUN: %optcall --collect-graph-stats --graph-stats-file=%t.json %s
// RUN: cat %t.json | FileCheck %s

// NOTE: Module taken from the "lospn-to-cpu" test "lower-to-cpu-structure-single.mlir"

module  {
  "lo_spn.kernel"() ( {
  ^bb0(%arg0: memref<?x2xi32>, %arg1: memref<?xf64>):  // no predecessors
    %c0 = constant 0 : index
    %0 = memref.dim %arg0, %c0 : memref<?x2xi32>
    %1 = memref.alloc(%0) : memref<?xf64>
    "lo_spn.task"(%arg0, %1) ( {
    ^bb0(%arg2: index, %arg3: memref<?x2xi32>, %arg4: memref<?xf64>):  // no predecessors
      %4 = "lo_spn.batch_read"(%arg3, %arg2) {sampleIndex = 0 : ui32} : (memref<?x2xi32>, index) -> i32
      %5 = "lo_spn.batch_read"(%arg3, %arg2) {sampleIndex = 1 : ui32} : (memref<?x2xi32>, index) -> i32
      %6 = "lo_spn.body"(%4, %5) ( {
      ^bb0(%arg5: i32, %arg6: i32):  // no predecessors
        %7 = "lo_spn.histogram"(%arg5) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 2.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 7.500000e-01 : f64}], supportMarginal = false} : (i32) -> f64
        %8 = "lo_spn.histogram"(%arg6) {bucketCount = 2 : ui32, buckets = [{lb = 0 : i32, ub = 1 : i32, val = 4.500000e-01 : f64}, {lb = 1 : i32, ub = 2 : i32, val = 5.500000e-01 : f64}], supportMarginal = false} : (i32) -> f64
        %9 = "lo_spn.mul"(%7, %8) : (f64, f64) -> f64
        %18 = "lo_spn.log"(%9) : (f64) -> f64
        "lo_spn.yield"(%18) : (f64) -> ()
      }) : (i32, i32) -> f64
      "lo_spn.batch_write"(%6, %arg4, %arg2) : (f64, memref<?xf64>, index) -> ()
      "lo_spn.return"() : () -> ()
    }) {batchSize = 1 : ui32} : (memref<?x2xi32>, memref<?xf64>) -> ()
    %2 = memref.tensor_load %1 : memref<?xf64>
    %3 = memref.buffer_cast %2 : memref<?xf64>
    "lo_spn.copy"(%3, %arg1) : (memref<?xf64>, memref<?xf64>) -> ()
    "lo_spn.return"() : () -> ()
  }) {sym_name = "spn_kernel", type = (memref<?x2xi32>, memref<?xf64>) -> ()} : () -> ()
}

// CHECK: {
// CHECK-DAG: "averageDepth":2.0
// CHECK-DAG: "categoricalCount":0
// CHECK-DAG: "constantCount":0
// CHECK-DAG: "featureCount":2
// CHECK-DAG: "gaussianCount":0
// CHECK-DAG: "histogramCount":2
// CHECK-DAG: "innerCount":1
// CHECK-DAG: "leafCount":2
// CHECK-DAG: "maxDepth":2
// CHECK-DAG: "medianDepth":1.5
// CHECK-DAG: "minDepth":2
// CHECK-DAG: "productCount":1
// CHECK-DAG: "sumCount":0
// CHECK: }
