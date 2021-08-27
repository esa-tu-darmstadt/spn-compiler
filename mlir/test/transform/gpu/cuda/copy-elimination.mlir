// RUN: %optcall --gpu-copy-elimination --gpu-kernel-outlining --gpu-buffer-deallocation %s | FileCheck %s


module  {
  func @spn_gpu(%arg0: memref<?x5xi32>, %arg1: memref<1x?xf64>) {
    %c0 = constant 0 : index
    %0 = memref.dim %arg0, %c0 : memref<?x5xi32>
    %1 = memref.alloc(%0) : memref<2x?xf64>
    %c0_0 = constant 0 : index
    %2 = memref.dim %arg0, %c0_0 : memref<?x5xi32>
    %memref = gpu.alloc  (%2) : memref<?x5xi32>
    %c1 = constant 1 : index
    %3 = memref.dim %1, %c1 : memref<2x?xf64>
    %memref_1 = gpu.alloc  (%3) : memref<2x?xf64>
    gpu.memcpy  %memref, %arg0 : memref<?x5xi32>, memref<?x5xi32>
    %c0_2 = constant 0 : index
    %4 = memref.dim %arg0, %c0_2 : memref<?x5xi32>
    %c64 = constant 64 : index
    %5 = ceildivi_signed %4, %c64 : index
    %c1_3 = constant 1 : index
    gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %5, %arg9 = %c1_3, %arg10 = %c1_3) threads(%arg5, %arg6, %arg7) in (%arg11 = %c64, %arg12 = %c1_3, %arg13 = %c1_3) {
      %17 = muli %c64, %arg2 : index
      %18 = addi %17, %arg5 : index
      %19 = cmpi ult, %18, %4 : index
      scf.if %19 {
        %20 = "lo_spn.batch_read"(%memref, %18) {staticIndex = 0 : ui32, transposed = false} : (memref<?x5xi32>, index) -> i32
        %21 = "lo_spn.batch_read"(%memref, %18) {staticIndex = 1 : ui32, transposed = false} : (memref<?x5xi32>, index) -> i32
        %22 = "lo_spn.batch_read"(%memref, %18) {staticIndex = 2 : ui32, transposed = false} : (memref<?x5xi32>, index) -> i32
        %23 = "lo_spn.categorical"(%20) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false} : (i32) -> f64
        %24 = "lo_spn.categorical"(%20) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false} : (i32) -> f64
        %25 = "lo_spn.categorical"(%21) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false} : (i32) -> f64
        %26 = "lo_spn.mul"(%23, %25) : (f64, f64) -> f64
        %27 = "lo_spn.categorical"(%21) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false} : (i32) -> f64
        %28 = "lo_spn.mul"(%24, %27) : (f64, f64) -> f64
        %29 = "lo_spn.categorical"(%22) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false} : (i32) -> f64
        %30 = "lo_spn.mul"(%26, %29) : (f64, f64) -> f64
        %31 = "lo_spn.categorical"(%22) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false} : (i32) -> f64
        %32 = "lo_spn.mul"(%28, %31) : (f64, f64) -> f64
        "lo_spn.batch_write"(%memref_1, %18, %30, %32) {transposed = true} : (memref<2x?xf64>, index, f64, f64) -> ()
      }
      gpu.terminator
    }
    gpu.memcpy  %1, %memref_1 : memref<2x?xf64>, memref<2x?xf64>
    %6 = memref.alloc(%0) : memref<2x?xf64>
    %c0_4 = constant 0 : index
    %7 = memref.dim %arg0, %c0_4 : memref<?x5xi32>
    %memref_5 = gpu.alloc  (%7) : memref<?x5xi32>
    %c1_6 = constant 1 : index
    %8 = memref.dim %1, %c1_6 : memref<2x?xf64>
    %memref_7 = gpu.alloc  (%8) : memref<2x?xf64>
    %c1_8 = constant 1 : index
    %9 = memref.dim %6, %c1_8 : memref<2x?xf64>
    %memref_9 = gpu.alloc  (%9) : memref<2x?xf64>
    gpu.memcpy  %memref_5, %arg0 : memref<?x5xi32>, memref<?x5xi32>
    gpu.memcpy  %memref_7, %1 : memref<2x?xf64>, memref<2x?xf64>
    %c0_10 = constant 0 : index
    %10 = memref.dim %arg0, %c0_10 : memref<?x5xi32>
    %c64_11 = constant 64 : index
    %11 = ceildivi_signed %10, %c64_11 : index
    %c1_12 = constant 1 : index
    gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %11, %arg9 = %c1_12, %arg10 = %c1_12) threads(%arg5, %arg6, %arg7) in (%arg11 = %c64_11, %arg12 = %c1_12, %arg13 = %c1_12) {
      %17 = muli %c64_11, %arg2 : index
      %18 = addi %17, %arg5 : index
      %19 = cmpi ult, %18, %10 : index
      scf.if %19 {
        %20 = "lo_spn.batch_read"(%memref_7, %18) {staticIndex = 1 : ui32, transposed = true} : (memref<2x?xf64>, index) -> f64
        %21 = "lo_spn.batch_read"(%memref_5, %18) {staticIndex = 3 : ui32, transposed = false} : (memref<?x5xi32>, index) -> i32
        %22 = "lo_spn.batch_read"(%memref_5, %18) {staticIndex = 4 : ui32, transposed = false} : (memref<?x5xi32>, index) -> i32
        %23 = "lo_spn.categorical"(%21) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false} : (i32) -> f64
        %24 = "lo_spn.categorical"(%21) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false} : (i32) -> f64
        %25 = "lo_spn.categorical"(%22) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false} : (i32) -> f64
        %26 = "lo_spn.mul"(%23, %25) : (f64, f64) -> f64
        %27 = "lo_spn.categorical"(%22) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false} : (i32) -> f64
        %28 = "lo_spn.mul"(%24, %27) : (f64, f64) -> f64
        %29 = "lo_spn.mul"(%20, %28) : (f64, f64) -> f64
        "lo_spn.batch_write"(%memref_9, %18, %26, %29) {transposed = true} : (memref<2x?xf64>, index, f64, f64) -> ()
      }
      gpu.terminator
    }
    gpu.memcpy  %6, %memref_9 : memref<2x?xf64>, memref<2x?xf64>
    %c1_13 = constant 1 : index
    %12 = memref.dim %1, %c1_13 : memref<2x?xf64>
    %memref_14 = gpu.alloc  (%12) : memref<2x?xf64>
    %c1_15 = constant 1 : index
    %13 = memref.dim %6, %c1_15 : memref<2x?xf64>
    %memref_16 = gpu.alloc  (%13) : memref<2x?xf64>
    %c1_17 = constant 1 : index
    %14 = memref.dim %arg1, %c1_17 : memref<1x?xf64>
    %memref_18 = gpu.alloc  (%14) : memref<1x?xf64>
    gpu.memcpy  %memref_14, %1 : memref<2x?xf64>, memref<2x?xf64>
    gpu.memcpy  %memref_16, %6 : memref<2x?xf64>, memref<2x?xf64>
    %c1_19 = constant 1 : index
    %15 = memref.dim %1, %c1_19 : memref<2x?xf64>
    %c64_20 = constant 64 : index
    %16 = ceildivi_signed %15, %c64_20 : index
    %c1_21 = constant 1 : index
    gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %16, %arg9 = %c1_21, %arg10 = %c1_21) threads(%arg5, %arg6, %arg7) in (%arg11 = %c64_20, %arg12 = %c1_21, %arg13 = %c1_21) {
      %17 = muli %c64_20, %arg2 : index
      %18 = addi %17, %arg5 : index
      %19 = cmpi ult, %18, %15 : index
      scf.if %19 {
        %20 = "lo_spn.batch_read"(%memref_14, %18) {staticIndex = 0 : ui32, transposed = true} : (memref<2x?xf64>, index) -> f64
        %21 = "lo_spn.batch_read"(%memref_16, %18) {staticIndex = 1 : ui32, transposed = true} : (memref<2x?xf64>, index) -> f64
        %22 = "lo_spn.batch_read"(%memref_16, %18) {staticIndex = 0 : ui32, transposed = true} : (memref<2x?xf64>, index) -> f64
        %cst = constant 0.80416666666666669 : f64
        %cst_22 = constant 0.19583333333333333 : f64
        %23 = "lo_spn.mul"(%20, %22) : (f64, f64) -> f64
        %24 = "lo_spn.mul"(%23, %cst) : (f64, f64) -> f64
        %25 = "lo_spn.mul"(%21, %cst_22) : (f64, f64) -> f64
        %26 = "lo_spn.add"(%25, %24) : (f64, f64) -> f64
        %27 = "lo_spn.log"(%26) : (f64) -> f64
        "lo_spn.batch_write"(%memref_18, %18, %27) {transposed = true} : (memref<1x?xf64>, index, f64) -> ()
      }
      gpu.terminator
    }
    gpu.memcpy  %arg1, %memref_18 : memref<1x?xf64>, memref<1x?xf64>
    "lo_spn.return"() : () -> ()
  }
}

// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py


// CHECK-LABEL:   func @spn_gpu(
// CHECK-SAME:                  %[[VAL_0:.*]]: memref<?x5xi32>,
// CHECK-SAME:                  %[[VAL_1:.*]]: memref<1x?xf64>) {
// CHECK:           %[[VAL_2:.*]] = constant 0 : index
// CHECK:           %[[VAL_3:.*]] = constant 64 : index
// CHECK:           %[[VAL_4:.*]] = constant 1 : index
// CHECK:           %[[VAL_5:.*]] = memref.dim %[[VAL_0]], %[[VAL_2]] : memref<?x5xi32>
// CHECK:           %[[VAL_6:.*]] = memref.dim %[[VAL_0]], %[[VAL_2]] : memref<?x5xi32>
// CHECK:           %[[VAL_7:.*]] = gpu.alloc  (%[[VAL_6]]) : memref<?x5xi32>
// CHECK:           %[[VAL_8:.*]] = gpu.alloc  (%[[VAL_5]]) : memref<2x?xf64>
// CHECK:           gpu.memcpy  %[[VAL_7]], %[[VAL_0]] : memref<?x5xi32>, memref<?x5xi32>
// CHECK:           %[[VAL_9:.*]] = memref.dim %[[VAL_0]], %[[VAL_2]] : memref<?x5xi32>
// CHECK:           %[[VAL_10:.*]] = ceildivi_signed %[[VAL_9]], %[[VAL_3]] : index
// CHECK:           gpu.launch_func  @spn_gpu_kernel::@spn_gpu_kernel blocks in (%[[VAL_10]], %[[VAL_4]], %[[VAL_4]]) threads in (%[[VAL_3]], %[[VAL_4]], %[[VAL_4]]) args(%[[VAL_9]] : index, %[[VAL_7]] : memref<?x5xi32>, %[[VAL_8]] : memref<2x?xf64>)
// CHECK:           %[[VAL_11:.*]] = gpu.alloc  (%[[VAL_5]]) : memref<2x?xf64>
// CHECK:           %[[VAL_12:.*]] = memref.dim %[[VAL_0]], %[[VAL_2]] : memref<?x5xi32>
// CHECK:           %[[VAL_13:.*]] = ceildivi_signed %[[VAL_12]], %[[VAL_3]] : index
// CHECK:           gpu.launch_func  @spn_gpu_kernel_0::@spn_gpu_kernel blocks in (%[[VAL_13]], %[[VAL_4]], %[[VAL_4]]) threads in (%[[VAL_3]], %[[VAL_4]], %[[VAL_4]]) args(%[[VAL_12]] : index, %[[VAL_8]] : memref<2x?xf64>, %[[VAL_7]] : memref<?x5xi32>, %[[VAL_11]] : memref<2x?xf64>)
// CHECK:           gpu.dealloc  %[[VAL_7]] : memref<?x5xi32>
// CHECK:           %[[VAL_14:.*]] = memref.dim %[[VAL_1]], %[[VAL_4]] : memref<1x?xf64>
// CHECK:           %[[VAL_15:.*]] = gpu.alloc  (%[[VAL_14]]) : memref<1x?xf64>
// CHECK:           %[[VAL_16:.*]] = ceildivi_signed %[[VAL_5]], %[[VAL_3]] : index
// CHECK:           gpu.launch_func  @spn_gpu_kernel_1::@spn_gpu_kernel blocks in (%[[VAL_16]], %[[VAL_4]], %[[VAL_4]]) threads in (%[[VAL_3]], %[[VAL_4]], %[[VAL_4]]) args(%[[VAL_5]] : index, %[[VAL_8]] : memref<2x?xf64>, %[[VAL_11]] : memref<2x?xf64>, %[[VAL_15]] : memref<1x?xf64>)
// CHECK:           gpu.dealloc  %[[VAL_8]] : memref<2x?xf64>
// CHECK:           gpu.dealloc  %[[VAL_11]] : memref<2x?xf64>
// CHECK:           gpu.memcpy  %[[VAL_1]], %[[VAL_15]] : memref<1x?xf64>, memref<1x?xf64>
// CHECK:           gpu.dealloc  %[[VAL_15]] : memref<1x?xf64>
// CHECK:           "lo_spn.return"() : () -> ()
// CHECK:         }

// CHECK-LABEL:   gpu.module @spn_gpu_kernel {
// CHECK:           gpu.func @spn_gpu_kernel(%[[VAL_0:.*]]: index, %[[VAL_1:.*]]: memref<?x5xi32>, %[[VAL_2:.*]]: memref<2x?xf64>) kernel {
// CHECK:             %[[VAL_3:.*]] = "gpu.block_id"() {dimension = "x"} : () -> index
// CHECK:             %[[VAL_4:.*]] = "gpu.block_id"() {dimension = "y"} : () -> index
// CHECK:             %[[VAL_5:.*]] = "gpu.block_id"() {dimension = "z"} : () -> index
// CHECK:             %[[VAL_6:.*]] = "gpu.thread_id"() {dimension = "x"} : () -> index
// CHECK:             %[[VAL_7:.*]] = "gpu.thread_id"() {dimension = "y"} : () -> index
// CHECK:             %[[VAL_8:.*]] = "gpu.thread_id"() {dimension = "z"} : () -> index
// CHECK:             %[[VAL_9:.*]] = "gpu.grid_dim"() {dimension = "x"} : () -> index
// CHECK:             %[[VAL_10:.*]] = "gpu.grid_dim"() {dimension = "y"} : () -> index
// CHECK:             %[[VAL_11:.*]] = "gpu.grid_dim"() {dimension = "z"} : () -> index
// CHECK:             %[[VAL_12:.*]] = "gpu.block_dim"() {dimension = "x"} : () -> index
// CHECK:             %[[VAL_13:.*]] = "gpu.block_dim"() {dimension = "y"} : () -> index
// CHECK:             %[[VAL_14:.*]] = "gpu.block_dim"() {dimension = "z"} : () -> index
// CHECK:             br ^bb1
// CHECK:           ^bb1:
// CHECK:             %[[VAL_15:.*]] = constant 64 : index
// CHECK:             %[[VAL_16:.*]] = muli %[[VAL_3]], %[[VAL_15]] : index
// CHECK:             %[[VAL_17:.*]] = addi %[[VAL_16]], %[[VAL_6]] : index
// CHECK:             %[[VAL_18:.*]] = cmpi ult, %[[VAL_17]], %[[VAL_0]] : index
// CHECK:             scf.if %[[VAL_18]] {
// CHECK:               %[[VAL_19:.*]] = "lo_spn.batch_read"(%[[VAL_1]], %[[VAL_17]]) {staticIndex = 0 : ui32, transposed = false} : (memref<?x5xi32>, index) -> i32
// CHECK:               %[[VAL_20:.*]] = "lo_spn.batch_read"(%[[VAL_1]], %[[VAL_17]]) {staticIndex = 1 : ui32, transposed = false} : (memref<?x5xi32>, index) -> i32
// CHECK:               %[[VAL_21:.*]] = "lo_spn.batch_read"(%[[VAL_1]], %[[VAL_17]]) {staticIndex = 2 : ui32, transposed = false} : (memref<?x5xi32>, index) -> i32
// CHECK:               %[[VAL_22:.*]] = "lo_spn.categorical"(%[[VAL_19]]) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false} : (i32) -> f64
// CHECK:               %[[VAL_23:.*]] = "lo_spn.categorical"(%[[VAL_19]]) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false} : (i32) -> f64
// CHECK:               %[[VAL_24:.*]] = "lo_spn.categorical"(%[[VAL_20]]) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false} : (i32) -> f64
// CHECK:               %[[VAL_25:.*]] = "lo_spn.mul"(%[[VAL_22]], %[[VAL_24]]) : (f64, f64) -> f64
// CHECK:               %[[VAL_26:.*]] = "lo_spn.categorical"(%[[VAL_20]]) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false} : (i32) -> f64
// CHECK:               %[[VAL_27:.*]] = "lo_spn.mul"(%[[VAL_23]], %[[VAL_26]]) : (f64, f64) -> f64
// CHECK:               %[[VAL_28:.*]] = "lo_spn.categorical"(%[[VAL_21]]) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false} : (i32) -> f64
// CHECK:               %[[VAL_29:.*]] = "lo_spn.mul"(%[[VAL_25]], %[[VAL_28]]) : (f64, f64) -> f64
// CHECK:               %[[VAL_30:.*]] = "lo_spn.categorical"(%[[VAL_21]]) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false} : (i32) -> f64
// CHECK:               %[[VAL_31:.*]] = "lo_spn.mul"(%[[VAL_27]], %[[VAL_30]]) : (f64, f64) -> f64
// CHECK:               "lo_spn.batch_write"(%[[VAL_2]], %[[VAL_17]], %[[VAL_29]], %[[VAL_31]]) {transposed = true} : (memref<2x?xf64>, index, f64, f64) -> ()
// CHECK:             }
// CHECK:             gpu.return
// CHECK:           }
// CHECK:         }

// CHECK-LABEL:   gpu.module @spn_gpu_kernel_0 {
// CHECK:           gpu.func @spn_gpu_kernel(%[[VAL_0:.*]]: index, %[[VAL_1:.*]]: memref<2x?xf64>, %[[VAL_2:.*]]: memref<?x5xi32>, %[[VAL_3:.*]]: memref<2x?xf64>) kernel {
// CHECK:             %[[VAL_4:.*]] = "gpu.block_id"() {dimension = "x"} : () -> index
// CHECK:             %[[VAL_5:.*]] = "gpu.block_id"() {dimension = "y"} : () -> index
// CHECK:             %[[VAL_6:.*]] = "gpu.block_id"() {dimension = "z"} : () -> index
// CHECK:             %[[VAL_7:.*]] = "gpu.thread_id"() {dimension = "x"} : () -> index
// CHECK:             %[[VAL_8:.*]] = "gpu.thread_id"() {dimension = "y"} : () -> index
// CHECK:             %[[VAL_9:.*]] = "gpu.thread_id"() {dimension = "z"} : () -> index
// CHECK:             %[[VAL_10:.*]] = "gpu.grid_dim"() {dimension = "x"} : () -> index
// CHECK:             %[[VAL_11:.*]] = "gpu.grid_dim"() {dimension = "y"} : () -> index
// CHECK:             %[[VAL_12:.*]] = "gpu.grid_dim"() {dimension = "z"} : () -> index
// CHECK:             %[[VAL_13:.*]] = "gpu.block_dim"() {dimension = "x"} : () -> index
// CHECK:             %[[VAL_14:.*]] = "gpu.block_dim"() {dimension = "y"} : () -> index
// CHECK:             %[[VAL_15:.*]] = "gpu.block_dim"() {dimension = "z"} : () -> index
// CHECK:             br ^bb1
// CHECK:           ^bb1:
// CHECK:             %[[VAL_16:.*]] = constant 64 : index
// CHECK:             %[[VAL_17:.*]] = muli %[[VAL_4]], %[[VAL_16]] : index
// CHECK:             %[[VAL_18:.*]] = addi %[[VAL_17]], %[[VAL_7]] : index
// CHECK:             %[[VAL_19:.*]] = cmpi ult, %[[VAL_18]], %[[VAL_0]] : index
// CHECK:             scf.if %[[VAL_19]] {
// CHECK:               %[[VAL_20:.*]] = "lo_spn.batch_read"(%[[VAL_1]], %[[VAL_18]]) {staticIndex = 1 : ui32, transposed = true} : (memref<2x?xf64>, index) -> f64
// CHECK:               %[[VAL_21:.*]] = "lo_spn.batch_read"(%[[VAL_2]], %[[VAL_18]]) {staticIndex = 3 : ui32, transposed = false} : (memref<?x5xi32>, index) -> i32
// CHECK:               %[[VAL_22:.*]] = "lo_spn.batch_read"(%[[VAL_2]], %[[VAL_18]]) {staticIndex = 4 : ui32, transposed = false} : (memref<?x5xi32>, index) -> i32
// CHECK:               %[[VAL_23:.*]] = "lo_spn.categorical"(%[[VAL_21]]) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false} : (i32) -> f64
// CHECK:               %[[VAL_24:.*]] = "lo_spn.categorical"(%[[VAL_21]]) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false} : (i32) -> f64
// CHECK:               %[[VAL_25:.*]] = "lo_spn.categorical"(%[[VAL_22]]) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false} : (i32) -> f64
// CHECK:               %[[VAL_26:.*]] = "lo_spn.mul"(%[[VAL_23]], %[[VAL_25]]) : (f64, f64) -> f64
// CHECK:               %[[VAL_27:.*]] = "lo_spn.categorical"(%[[VAL_22]]) {probabilities = [3.500000e-01, 5.500000e-01, 1.000000e-01], supportMarginal = false} : (i32) -> f64
// CHECK:               %[[VAL_28:.*]] = "lo_spn.mul"(%[[VAL_24]], %[[VAL_27]]) : (f64, f64) -> f64
// CHECK:               %[[VAL_29:.*]] = "lo_spn.mul"(%[[VAL_20]], %[[VAL_28]]) : (f64, f64) -> f64
// CHECK:               "lo_spn.batch_write"(%[[VAL_3]], %[[VAL_18]], %[[VAL_26]], %[[VAL_29]]) {transposed = true} : (memref<2x?xf64>, index, f64, f64) -> ()
// CHECK:             }
// CHECK:             gpu.return
// CHECK:           }
// CHECK:         }

// CHECK-LABEL:   gpu.module @spn_gpu_kernel_1 {
// CHECK:           gpu.func @spn_gpu_kernel(%[[VAL_0:.*]]: index, %[[VAL_1:.*]]: memref<2x?xf64>, %[[VAL_2:.*]]: memref<2x?xf64>, %[[VAL_3:.*]]: memref<1x?xf64>) kernel {
// CHECK:             %[[VAL_4:.*]] = "gpu.block_id"() {dimension = "x"} : () -> index
// CHECK:             %[[VAL_5:.*]] = "gpu.block_id"() {dimension = "y"} : () -> index
// CHECK:             %[[VAL_6:.*]] = "gpu.block_id"() {dimension = "z"} : () -> index
// CHECK:             %[[VAL_7:.*]] = "gpu.thread_id"() {dimension = "x"} : () -> index
// CHECK:             %[[VAL_8:.*]] = "gpu.thread_id"() {dimension = "y"} : () -> index
// CHECK:             %[[VAL_9:.*]] = "gpu.thread_id"() {dimension = "z"} : () -> index
// CHECK:             %[[VAL_10:.*]] = "gpu.grid_dim"() {dimension = "x"} : () -> index
// CHECK:             %[[VAL_11:.*]] = "gpu.grid_dim"() {dimension = "y"} : () -> index
// CHECK:             %[[VAL_12:.*]] = "gpu.grid_dim"() {dimension = "z"} : () -> index
// CHECK:             %[[VAL_13:.*]] = "gpu.block_dim"() {dimension = "x"} : () -> index
// CHECK:             %[[VAL_14:.*]] = "gpu.block_dim"() {dimension = "y"} : () -> index
// CHECK:             %[[VAL_15:.*]] = "gpu.block_dim"() {dimension = "z"} : () -> index
// CHECK:             br ^bb1
// CHECK:           ^bb1:
// CHECK:             %[[VAL_16:.*]] = constant 64 : index
// CHECK:             %[[VAL_17:.*]] = constant 0.80416666666666669 : f64
// CHECK:             %[[VAL_18:.*]] = constant 0.19583333333333333 : f64
// CHECK:             %[[VAL_19:.*]] = muli %[[VAL_4]], %[[VAL_16]] : index
// CHECK:             %[[VAL_20:.*]] = addi %[[VAL_19]], %[[VAL_7]] : index
// CHECK:             %[[VAL_21:.*]] = cmpi ult, %[[VAL_20]], %[[VAL_0]] : index
// CHECK:             scf.if %[[VAL_21]] {
// CHECK:               %[[VAL_22:.*]] = "lo_spn.batch_read"(%[[VAL_1]], %[[VAL_20]]) {staticIndex = 0 : ui32, transposed = true} : (memref<2x?xf64>, index) -> f64
// CHECK:               %[[VAL_23:.*]] = "lo_spn.batch_read"(%[[VAL_2]], %[[VAL_20]]) {staticIndex = 1 : ui32, transposed = true} : (memref<2x?xf64>, index) -> f64
// CHECK:               %[[VAL_24:.*]] = "lo_spn.batch_read"(%[[VAL_2]], %[[VAL_20]]) {staticIndex = 0 : ui32, transposed = true} : (memref<2x?xf64>, index) -> f64
// CHECK:               %[[VAL_25:.*]] = "lo_spn.mul"(%[[VAL_22]], %[[VAL_24]]) : (f64, f64) -> f64
// CHECK:               %[[VAL_26:.*]] = "lo_spn.mul"(%[[VAL_25]], %[[VAL_17]]) : (f64, f64) -> f64
// CHECK:               %[[VAL_27:.*]] = "lo_spn.mul"(%[[VAL_23]], %[[VAL_18]]) : (f64, f64) -> f64
// CHECK:               %[[VAL_28:.*]] = "lo_spn.add"(%[[VAL_27]], %[[VAL_26]]) : (f64, f64) -> f64
// CHECK:               %[[VAL_29:.*]] = "lo_spn.log"(%[[VAL_28]]) : (f64) -> f64
// CHECK:               "lo_spn.batch_write"(%[[VAL_3]], %[[VAL_20]], %[[VAL_29]]) {transposed = true} : (memref<1x?xf64>, index, f64) -> ()
// CHECK:             }
// CHECK:             gpu.return
// CHECK:           }
// CHECK:         }

