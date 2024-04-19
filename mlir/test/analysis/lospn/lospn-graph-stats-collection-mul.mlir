// RUN: %optcall --collect-graph-stats --graph-stats-file=%t.json %s
// RUN: cat %t.json | FileCheck %s

// NOTE: Module taken from the "lospn-to-cpu" test "lower-to-cpu-structure-single.mlir"

module  {
  "lo_spn.kernel"() ( {
  ^bb0(%arg0: memref<?x2xi32>, %arg1: memref<1x?xf64>):  // no predecessors
    %c0 = arith.constant 0 : index
    %0 = memref.dim %arg0, %c0 : memref<?x2xi32>
    %1 = memref.alloc(%0) : memref<1x?xf64>
    "lo_spn.task"(%arg0, %1) ( {
    ^bb0(%arg2: index, %arg3: memref<?x2xi32>, %arg4: memref<1x?xf64>):  // no predecessors
      %4 = "lo_spn.batch_read"(%arg3, %arg2) {staticIndex = 0 : ui32} : (memref<?x2xi32>, index) -> i32
      %5 = "lo_spn.batch_read"(%arg3, %arg2) {staticIndex = 1 : ui32} : (memref<?x2xi32>, index) -> i32
      %6 = "lo_spn.body"(%4, %5) ( {
      ^bb0(%arg5: i32, %arg6: i32):  // no predecessors
        %7 = "lo_spn.histogram"(%arg5) {bucketCount = 2 : ui32, buckets = [#lo_spn.bucket<0, 1, 2.500000e-01>, #lo_spn.bucket<1, 2, 7.500000e-01>], supportMarginal = false} : (i32) -> f64
        %8 = "lo_spn.histogram"(%arg6) {bucketCount = 2 : ui32, buckets = [#lo_spn.bucket<0, 1, 4.500000e-01>, #lo_spn.bucket<1, 2, 5.500000e-01>], supportMarginal = false} : (i32) -> f64
        %9 = "lo_spn.mul"(%7, %8) : (f64, f64) -> f64
        %18 = "lo_spn.log"(%9) : (f64) -> f64
        "lo_spn.yield"(%18) : (f64) -> ()
      }) : (i32, i32) -> f64
      "lo_spn.batch_write"(%arg4, %arg2, %6) {transposed = true} : (memref<1x?xf64>, index, f64) -> ()
      "lo_spn.return"() : () -> ()
    }) {batchSize = 1 : ui32} : (memref<?x2xi32>, memref<1x?xf64>) -> ()
    %2 = bufferization.to_tensor %1 : memref<1x?xf64>
    %3 = bufferization.to_memref %2 : memref<1x?xf64>
    "lo_spn.copy"(%3, %arg1) : (memref<1x?xf64>, memref<1x?xf64>) -> ()
    "lo_spn.return"() : () -> ()
  }) {sym_name = "spn_kernel", function_type = (memref<?x2xi32>, memref<1x?xf64>) -> ()} : () -> ()
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
