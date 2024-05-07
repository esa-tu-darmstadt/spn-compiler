// RUN: %optcall --gpu-copy-elimination --gpu-kernel-outlining --gpu-buffer-deallocation %s | FileCheck %s
module {
  func.func @spn_gpu(%arg0: memref<?x5xi32>, %arg1: memref<1x?xf64>) {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x5xi32>
    %alloc = memref.alloc(%dim) : memref<2x?xf64>
    %c0_0 = arith.constant 0 : index
    %dim_1 = memref.dim %arg0, %c0_0 : memref<?x5xi32>
    %memref = gpu.alloc  (%dim_1) : memref<?x5xi32>
    %c1 = arith.constant 1 : index
    %dim_2 = memref.dim %alloc, %c1 : memref<2x?xf64>
    %memref_3 = gpu.alloc  (%dim_2) : memref<2x?xf64>
    gpu.memcpy  %memref, %arg0 : memref<?x5xi32>, memref<?x5xi32>
    %c0_4 = arith.constant 0 : index
    %dim_5 = memref.dim %arg0, %c0_4 : memref<?x5xi32>
    %c64 = arith.constant 64 : index
    %0 = arith.remui %dim_5, %c64 : index
    %c1_6 = arith.constant 1 : index
    gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %0, %arg9 = %c1_6, %arg10 = %c1_6) threads(%arg5, %arg6, %arg7) in (%arg11 = %c64, %arg12 = %c1_6, %arg13 = %c1_6) {
      %3 = arith.muli %c64, %arg2 : index
      %4 = arith.addi %3, %arg5 : index
      %5 = arith.cmpi ult, %4, %dim_5 : index
      scf.if %5 {
        %6 = "lo_spn.batch_read"(%memref, %4) <{staticIndex = 0 : ui32, transposed = false}> : (memref<?x5xi32>, index) -> i32
        %7 = "lo_spn.batch_read"(%memref, %4) <{staticIndex = 1 : ui32, transposed = false}> : (memref<?x5xi32>, index) -> i32
        %8 = "lo_spn.batch_read"(%memref, %4) <{staticIndex = 2 : ui32, transposed = false}> : (memref<?x5xi32>, index) -> i32
        %9 = "lo_spn.categorical"(%6) <{probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false}> : (i32) -> f64
        %10 = "lo_spn.categorical"(%6) <{probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false}> : (i32) -> f64
        %11 = "lo_spn.categorical"(%7) <{probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false}> : (i32) -> f64
        %12 = "lo_spn.mul"(%9, %11) : (f64, f64) -> f64
        %13 = "lo_spn.categorical"(%7) <{probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false}> : (i32) -> f64
        %14 = "lo_spn.mul"(%10, %13) : (f64, f64) -> f64
        %15 = "lo_spn.categorical"(%8) <{probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false}> : (i32) -> f64
        %16 = "lo_spn.mul"(%12, %15) : (f64, f64) -> f64
        %17 = "lo_spn.categorical"(%8) <{probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false}> : (i32) -> f64
        %18 = "lo_spn.mul"(%14, %17) : (f64, f64) -> f64
        "lo_spn.batch_write"(%memref_3, %4, %16, %18) <{transposed = true}> : (memref<2x?xf64>, index, f64, f64) -> ()
      }
      gpu.terminator
    }
    gpu.memcpy  %alloc, %memref_3 : memref<2x?xf64>, memref<2x?xf64>
    %alloc_7 = memref.alloc(%dim) : memref<2x?xf64>
    %c0_8 = arith.constant 0 : index
    %dim_9 = memref.dim %arg0, %c0_8 : memref<?x5xi32>
    %memref_10 = gpu.alloc  (%dim_9) : memref<?x5xi32>
    %c1_11 = arith.constant 1 : index
    %dim_12 = memref.dim %alloc, %c1_11 : memref<2x?xf64>
    %memref_13 = gpu.alloc  (%dim_12) : memref<2x?xf64>
    %c1_14 = arith.constant 1 : index
    %dim_15 = memref.dim %alloc_7, %c1_14 : memref<2x?xf64>
    %memref_16 = gpu.alloc  (%dim_15) : memref<2x?xf64>
    gpu.memcpy  %memref_10, %arg0 : memref<?x5xi32>, memref<?x5xi32>
    gpu.memcpy  %memref_13, %alloc : memref<2x?xf64>, memref<2x?xf64>
    %c0_17 = arith.constant 0 : index
    %dim_18 = memref.dim %arg0, %c0_17 : memref<?x5xi32>
    %c64_19 = arith.constant 64 : index
    %1 = arith.remui %dim_18, %c64_19 : index
    %c1_20 = arith.constant 1 : index
    gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %1, %arg9 = %c1_20, %arg10 = %c1_20) threads(%arg5, %arg6, %arg7) in (%arg11 = %c64_19, %arg12 = %c1_20, %arg13 = %c1_20) {
      %3 = arith.muli %c64_19, %arg2 : index
      %4 = arith.addi %3, %arg5 : index
      %5 = arith.cmpi ult, %4, %dim_18 : index
      scf.if %5 {
        %6 = "lo_spn.batch_read"(%memref_13, %4) <{staticIndex = 1 : ui32, transposed = true}> : (memref<2x?xf64>, index) -> f64
        %7 = "lo_spn.batch_read"(%memref_10, %4) <{staticIndex = 3 : ui32, transposed = false}> : (memref<?x5xi32>, index) -> i32
        %8 = "lo_spn.batch_read"(%memref_10, %4) <{staticIndex = 4 : ui32, transposed = false}> : (memref<?x5xi32>, index) -> i32
        %9 = "lo_spn.categorical"(%7) <{probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false}> : (i32) -> f64
        %10 = "lo_spn.categorical"(%7) <{probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false}> : (i32) -> f64
        %11 = "lo_spn.categorical"(%8) <{probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false}> : (i32) -> f64
        %12 = "lo_spn.mul"(%9, %11) : (f64, f64) -> f64
        %13 = "lo_spn.categorical"(%8) <{probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false}> : (i32) -> f64
        %14 = "lo_spn.mul"(%10, %13) : (f64, f64) -> f64
        %15 = "lo_spn.mul"(%6, %14) : (f64, f64) -> f64
        "lo_spn.batch_write"(%memref_16, %4, %12, %15) <{transposed = true}> : (memref<2x?xf64>, index, f64, f64) -> ()
      }
      gpu.terminator
    }
    gpu.memcpy  %alloc_7, %memref_16 : memref<2x?xf64>, memref<2x?xf64>
    %c1_21 = arith.constant 1 : index
    %dim_22 = memref.dim %alloc, %c1_21 : memref<2x?xf64>
    %memref_23 = gpu.alloc  (%dim_22) : memref<2x?xf64>
    %c1_24 = arith.constant 1 : index
    %dim_25 = memref.dim %alloc_7, %c1_24 : memref<2x?xf64>
    %memref_26 = gpu.alloc  (%dim_25) : memref<2x?xf64>
    %c1_27 = arith.constant 1 : index
    %dim_28 = memref.dim %arg1, %c1_27 : memref<1x?xf64>
    %memref_29 = gpu.alloc  (%dim_28) : memref<1x?xf64>
    gpu.memcpy  %memref_23, %alloc : memref<2x?xf64>, memref<2x?xf64>
    gpu.memcpy  %memref_26, %alloc_7 : memref<2x?xf64>, memref<2x?xf64>
    %c1_30 = arith.constant 1 : index
    %dim_31 = memref.dim %alloc, %c1_30 : memref<2x?xf64>
    %c64_32 = arith.constant 64 : index
    %2 = arith.remui %dim_31, %c64_32 : index
    %c1_33 = arith.constant 1 : index
    gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %2, %arg9 = %c1_33, %arg10 = %c1_33) threads(%arg5, %arg6, %arg7) in (%arg11 = %c64_32, %arg12 = %c1_33, %arg13 = %c1_33) {
      %3 = arith.muli %c64_32, %arg2 : index
      %4 = arith.addi %3, %arg5 : index
      %5 = arith.cmpi ult, %4, %dim_31 : index
      scf.if %5 {
        %6 = "lo_spn.batch_read"(%memref_23, %4) <{staticIndex = 0 : ui32, transposed = true}> : (memref<2x?xf64>, index) -> f64
        %7 = "lo_spn.batch_read"(%memref_26, %4) <{staticIndex = 1 : ui32, transposed = true}> : (memref<2x?xf64>, index) -> f64
        %8 = "lo_spn.batch_read"(%memref_26, %4) <{staticIndex = 0 : ui32, transposed = true}> : (memref<2x?xf64>, index) -> f64
        %cst = arith.constant 0.80416666666666669 : f64
        %cst_34 = arith.constant 0.19583333333333333 : f64
        %9 = "lo_spn.mul"(%6, %8) : (f64, f64) -> f64
        %10 = "lo_spn.mul"(%9, %cst) : (f64, f64) -> f64
        %11 = "lo_spn.mul"(%7, %cst_34) : (f64, f64) -> f64
        %12 = "lo_spn.add"(%11, %10) : (f64, f64) -> f64
        %13 = "lo_spn.log"(%12) : (f64) -> f64
        "lo_spn.batch_write"(%memref_29, %4, %13) <{transposed = true}> : (memref<1x?xf64>, index, f64) -> ()
      }
      gpu.terminator
    }
    gpu.memcpy  %arg1, %memref_29 : memref<1x?xf64>, memref<1x?xf64>
    "lo_spn.return"() : () -> ()
  }
}

