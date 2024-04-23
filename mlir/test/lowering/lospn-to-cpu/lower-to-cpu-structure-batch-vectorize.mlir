// RUN: %optcall --convert-lospn-structure-to-cpu --cpu-vectorize=true %s | FileCheck %s
module {
  "lo_spn.kernel"() <{function_type = (memref<?x6xf64>, memref<1x?xf64>) -> (), sym_name = "spn_cpu"}> ({
  ^bb0(%arg0: memref<?x6xf64>, %arg1: memref<1x?xf64>):
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x6xf64>
    %alloc = memref.alloc(%dim) : memref<1x?xf64>
    "lo_spn.task"(%arg0, %alloc) <{batchSize = 12 : ui32}> ({
    ^bb0(%arg2: index, %arg3: memref<?x6xf64>, %arg4: memref<1x?xf64>):
      %0 = "lo_spn.batch_read"(%arg3, %arg2) <{staticIndex = 0 : ui32}> : (memref<?x6xf64>, index) -> f64
      %1 = "lo_spn.batch_read"(%arg3, %arg2) <{staticIndex = 1 : ui32}> : (memref<?x6xf64>, index) -> f64
      %2 = "lo_spn.batch_read"(%arg3, %arg2) <{staticIndex = 2 : ui32}> : (memref<?x6xf64>, index) -> f64
      %3 = "lo_spn.batch_read"(%arg3, %arg2) <{staticIndex = 3 : ui32}> : (memref<?x6xf64>, index) -> f64
      %4 = "lo_spn.batch_read"(%arg3, %arg2) <{staticIndex = 4 : ui32}> : (memref<?x6xf64>, index) -> f64
      %5 = "lo_spn.batch_read"(%arg3, %arg2) <{staticIndex = 5 : ui32}> : (memref<?x6xf64>, index) -> f64
      %6 = "lo_spn.body"(%0, %1, %2, %3, %4, %5) ({
      ^bb0(%arg5: f64, %arg6: f64, %arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64):
        %7 = "lo_spn.categorical"(%arg5) <{probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false}> : (f64) -> f64
        %8 = "lo_spn.categorical"(%arg6) <{probabilities = [2.500000e-01, 6.250000e-01, 1.250000e-01], supportMarginal = false}> : (f64) -> f64
        %9 = "lo_spn.histogram"(%arg7) <{bucketCount = 2 : ui32, buckets = [#hi_spn.bucket<0 to 1 = 2.500000e-01>, #hi_spn.bucket<1 to 2 = 7.500000e-01>], supportMarginal = false}> : (f64) -> f64
        %10 = "lo_spn.histogram"(%arg8) <{bucketCount = 2 : ui32, buckets = [#hi_spn.bucket<0 to 1 = 4.500000e-01>, #hi_spn.bucket<1 to 2 = 5.500000e-01>], supportMarginal = false}> : (f64) -> f64
        %11 = "lo_spn.gaussian"(%arg9) <{mean = 5.000000e-01 : f64, stddev = 1.000000e+00 : f64, supportMarginal = false}> : (f64) -> f64
        %12 = "lo_spn.gaussian"(%arg10) <{mean = 2.500000e-01 : f64, stddev = 1.000000e-01 : f64, supportMarginal = false}> : (f64) -> f64
        %13 = "lo_spn.mul"(%7, %8) : (f64, f64) -> f64
        %14 = "lo_spn.mul"(%13, %9) : (f64, f64) -> f64
        %15 = "lo_spn.constant"() <{type = f64, value = 1.000000e-01 : f64}> : () -> f64
        %16 = "lo_spn.mul"(%14, %15) : (f64, f64) -> f64
        %17 = "lo_spn.mul"(%10, %11) : (f64, f64) -> f64
        %18 = "lo_spn.mul"(%17, %12) : (f64, f64) -> f64
        %19 = "lo_spn.constant"() <{type = f64, value = 1.000000e-01 : f64}> : () -> f64
        %20 = "lo_spn.mul"(%18, %19) : (f64, f64) -> f64
        %21 = "lo_spn.add"(%16, %20) : (f64, f64) -> f64
        %22 = "lo_spn.log"(%21) : (f64) -> f64
        "lo_spn.yield"(%22) : (f64) -> ()
      }) : (f64, f64, f64, f64, f64, f64) -> f64
      "lo_spn.batch_write"(%arg4, %arg2, %6) <{transposed = true}> : (memref<1x?xf64>, index, f64) -> ()
      "lo_spn.return"() : () -> ()
    }) : (memref<?x6xf64>, memref<1x?xf64>) -> ()
    "lo_spn.copy"(%alloc, %arg1) : (memref<1x?xf64>, memref<1x?xf64>) -> ()
    "lo_spn.return"() : () -> ()
  }) {type = (memref<?x6xf64>, memref<1x?xf64>) -> ()} : () -> ()
}

