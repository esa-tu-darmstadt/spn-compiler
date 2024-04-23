// RUN: %optcall --collect-graph-stats --graph-stats-file=%t.json %s
// RUN: cat %t.json | FileCheck %s

// NOTE: Module taken from the test "lowering/hispn-to-lospn/lower-to-lospn-sum.mlir"

module  {
  "lo_spn.kernel"() <{function_type = (tensor<?x5xi32>) -> (tensor<1x?xf64>), sym_name = "spn_cpu"}> ({
  ^bb0(%arg0: tensor<?x5xi32>):  // no predecessors
    %0 = "lo_spn.task"(%arg0) ( {
    ^bb0(%arg1: index, %arg2: tensor<?x5xi32>):  // no predecessors
      %1 = "lo_spn.batch_extract"(%arg2, %arg1) <{staticIndex = 0 : ui32}> : (tensor<?x5xi32>, index) -> i32
      %2 = "lo_spn.batch_extract"(%arg2, %arg1) <{staticIndex = 1 : ui32}> : (tensor<?x5xi32>, index) -> i32
      %3 = "lo_spn.batch_extract"(%arg2, %arg1) <{staticIndex = 2 : ui32}> : (tensor<?x5xi32>, index) -> i32
      %4 = "lo_spn.batch_extract"(%arg2, %arg1) <{staticIndex = 3 : ui32}> : (tensor<?x5xi32>, index) -> i32
      %5 = "lo_spn.batch_extract"(%arg2, %arg1) <{staticIndex = 4 : ui32}> : (tensor<?x5xi32>, index) -> i32
      %6 = "lo_spn.body"(%1, %2, %3, %4, %5) ( {
      ^bb0(%arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32):  // no predecessors
        %8 = "lo_spn.categorical"(%arg3) <{probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false}> : (i32) -> f64
        %9 = "lo_spn.categorical"(%arg4) <{probabilities = [2.500000e-01, 6.250000e-01, 1.250000e-01], supportMarginal = false}> : (i32) -> f64
        %10 = "lo_spn.categorical"(%arg5) <{probabilities = [5.000000e-01, 2.000000e-01, 3.000000e-01], supportMarginal = false}> : (i32) -> f64
        %11 = "lo_spn.categorical"(%arg6) <{probabilities = [6.000000e-01, 1.500000e-01, 2.500000e-01], supportMarginal = false}> : (i32) -> f64
        %12 = "lo_spn.categorical"(%arg7) <{probabilities = [8.000000e-01, 1.400000e-01, 6.000000e-02], supportMarginal = false}> : (i32) -> f64
        %13 = "lo_spn.constant"() <{type = f64, value = 1.000000e-01 : f64}> : () -> f64
        %14 = "lo_spn.mul"(%8, %13) : (f64, f64) -> f64
        %15 = "lo_spn.constant"() <{type = f64, value = 1.000000e-01 : f64}> : () -> f64
        %16 = "lo_spn.mul"(%9, %15) : (f64, f64) -> f64
        %17 = "lo_spn.add"(%14, %16) : (f64, f64) -> f64
        %18 = "lo_spn.constant"() <{type = f64, value = 1.000000e-01 : f64}> : () -> f64
        %19 = "lo_spn.mul"(%10, %18) : (f64, f64) -> f64
        %20 = "lo_spn.add"(%17, %19) : (f64, f64) -> f64
        %21 = "lo_spn.constant"() <{type = f64, value = 4.000000e-01 : f64}> : () -> f64
        %22 = "lo_spn.mul"(%11, %21) : (f64, f64) -> f64
        %23 = "lo_spn.constant"() <{type = f64, value = 3.000000e-01 : f64}> : () -> f64
        %24 = "lo_spn.mul"(%12, %23) : (f64, f64) -> f64
        %25 = "lo_spn.add"(%22, %24) : (f64, f64) -> f64
        %26 = "lo_spn.add"(%20, %25) : (f64, f64) -> f64
        %27 = "lo_spn.log"(%26) : (f64) -> f64
        "lo_spn.yield"(%27) : (f64) -> ()
      }) : (i32, i32, i32, i32, i32) -> f64
      %7 = "lo_spn.batch_collect"(%arg1, %6) {transposed=true} : (index, f64) -> tensor<1x?xf64>
      "lo_spn.return"(%7) : (tensor<1x?xf64>) -> ()
    }) {batchSize = 36 : ui32} : (tensor<?x5xi32>) -> tensor<1x?xf64>
    "lo_spn.return"(%0) : (tensor<1x?xf64>) -> ()
  }) {sym_name = "spn_kernel", function_type = (tensor<?x5xi32>) -> tensor<1x?xf64>} : () -> ()
}

// CHECK: {
// CHECK-DAG: "averageDepth":4.4
// CHECK-DAG: "categoricalCount":5
// CHECK-DAG: "constantCount":5
// CHECK-DAG: "featureCount":5
// CHECK-DAG: "gaussianCount":0
// CHECK-DAG: "histogramCount":0
// CHECK-DAG: "innerCount":9
// CHECK-DAG: "leafCount":10
// CHECK-DAG: "maxDepth":5
// CHECK-DAG: "medianDepth":4
// CHECK-DAG: "minDepth":4
// CHECK-DAG: "productCount":5
// CHECK-DAG: "sumCount":4
// CHECK: }
