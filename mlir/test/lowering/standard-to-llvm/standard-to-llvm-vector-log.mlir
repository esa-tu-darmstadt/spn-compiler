// RUN: %optcall --convert-math-to-llvm --convert-memref-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --convert-cf-to-llvm --convert-vector-to-llvm %s | FileCheck %s

module  {
  func.func @vec_task_0(%arg0: memref<?x4xf64>, %arg1: memref<?xf32>) {
    %c8 = arith.constant 8 : index
    %cst = arith.constant dense<[0, 4, 8, 12, 16, 20, 24, 28]> : vector<8xi64>
    %cst_0 = arith.constant dense<[1, 5, 9, 13, 17, 21, 25, 29]> : vector<8xi64>
    %cst_1 = arith.constant dense<[2, 6, 10, 14, 18, 22, 26, 30]> : vector<8xi64>
    %cst_2 = arith.constant dense<[3, 7, 11, 15, 19, 23, 27, 31]> : vector<8xi64>
    %cst_3 = arith.constant dense<4> : vector<8xi64>
    %cst_4 = arith.constant dense<0.000000e+00> : vector<8xf64>
    %cst_5 = arith.constant dense<true> : vector<8xi1>
    %c4 = arith.constant 4 : index
    %cst_6 = arith.constant dense<-5.000000e-01> : vector<8xf32>
    %cst_7 = arith.constant dense<-0.918938517> : vector<8xf32>
    %cst_8 = arith.constant dense<1.100000e-01> : vector<8xf32>
    %cst_9 = arith.constant dense<-0.888888895> : vector<8xf32>
    %cst_10 = arith.constant dense<-0.631256461> : vector<8xf32>
    %cst_11 = arith.constant dense<1.200000e-01> : vector<8xf32>
    %cst_12 = arith.constant dense<-2.000000e+00> : vector<8xf32>
    %cst_13 = arith.constant dense<-0.22579135> : vector<8xf32>
    %cst_14 = arith.constant dense<1.300000e-01> : vector<8xf32>
    %cst_15 = arith.constant dense<-8.000000e+00> : vector<8xf32>
    %cst_16 = arith.constant dense<0.467355818> : vector<8xf32>
    %cst_17 = arith.constant dense<1.400000e-01> : vector<8xf32>
    %cst_18 = arith.constant dense<-1.38629436> : vector<8xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %cst_19 = arith.constant -5.000000e-01 : f32
    %cst_20 = arith.constant -0.918938517 : f32
    %cst_21 = arith.constant 1.100000e-01 : f32
    %cst_22 = arith.constant -0.888888895 : f32
    %cst_23 = arith.constant -0.631256461 : f32
    %cst_24 = arith.constant 1.200000e-01 : f32
    %cst_25 = arith.constant -2.000000e+00 : f32
    %cst_26 = arith.constant -0.22579135 : f32
    %cst_27 = arith.constant 1.300000e-01 : f32
    %cst_28 = arith.constant -8.000000e+00 : f32
    %cst_29 = arith.constant 0.467355818 : f32
    %cst_30 = arith.constant 1.400000e-01 : f32
    %cst_31 = arith.constant -1.38629436 : f32
    %0 = memref.dim %arg0, %c0 : memref<?x4xf64>
    %1 = arith.remui %0, %c8 : index
    %2 = arith.subi %0, %1 : index
    cf.br ^bb1(%c0 : index)
  ^bb1(%3: index):  // 2 preds: ^bb0, ^bb2
    %4 = arith.cmpi slt, %3, %2 : index
    cf.cond_br %4, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %5 = arith.index_cast %3 : index to i64
    %6 = vector.splat %5 : vector<8xi64>
    %7 = arith.muli %6, %cst_3 : vector<8xi64>
    %8 = arith.addi %7, %cst : vector<8xi64>
    %9 = memref.dim %arg0, %c0 : memref<?x4xf64>
    %10 = arith.muli %9, %c4 : index
    %11 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%10], strides: [1] : memref<?x4xf64> to memref<?xf64>
    %12 = builtin.unrealized_conversion_cast %11 : memref<?xf64> to !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %13 = builtin.unrealized_conversion_cast %c0 : index to i64
    %14 = llvm.extractvalue %12[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %15 = llvm.getelementptr %14[%13] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %16 = llvm.getelementptr %15[%8] : (!llvm.ptr<f64>, vector<8xi64>) -> !llvm.vec<8 x ptr<f64>>
    %17 = llvm.intr.masked.gather %16, %cst_5, %cst_4 {alignment = 8 : i32} : (!llvm.vec<8 x ptr<f64>>, vector<8xi1>, vector<8xf64>) -> vector<8xf64>
    %18 = arith.index_cast %3 : index to i64
    %19 = vector.splat %18 : vector<8xi64>
    %20 = arith.muli %19, %cst_3 : vector<8xi64>
    %21 = arith.addi %20, %cst_0 : vector<8xi64>
    %22 = memref.dim %arg0, %c0 : memref<?x4xf64>
    %23 = arith.muli %22, %c4 : index
    %24 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%23], strides: [1] : memref<?x4xf64> to memref<?xf64>
    %25 = builtin.unrealized_conversion_cast %24 : memref<?xf64> to !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %26 = builtin.unrealized_conversion_cast %c0 : index to i64
    %27 = llvm.extractvalue %25[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %28 = llvm.getelementptr %27[%26] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %29 = llvm.getelementptr %28[%21] : (!llvm.ptr<f64>, vector<8xi64>) -> !llvm.vec<8 x ptr<f64>>
    %30 = llvm.intr.masked.gather %29, %cst_5, %cst_4 {alignment = 8 : i32} : (!llvm.vec<8 x ptr<f64>>, vector<8xi1>, vector<8xf64>) -> vector<8xf64>
    %31 = arith.index_cast %3 : index to i64
    %32 = vector.splat %31 : vector<8xi64>
    %33 = arith.muli %32, %cst_3 : vector<8xi64>
    %34 = arith.addi %33, %cst_1 : vector<8xi64>
    %35 = memref.dim %arg0, %c0 : memref<?x4xf64>
    %36 = arith.muli %35, %c4 : index
    %37 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%36], strides: [1] : memref<?x4xf64> to memref<?xf64>
    %38 = builtin.unrealized_conversion_cast %37 : memref<?xf64> to !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %39 = builtin.unrealized_conversion_cast %c0 : index to i64
    %40 = llvm.extractvalue %38[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %41 = llvm.getelementptr %40[%39] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %42 = llvm.getelementptr %41[%34] : (!llvm.ptr<f64>, vector<8xi64>) -> !llvm.vec<8 x ptr<f64>>
    %43 = llvm.intr.masked.gather %42, %cst_5, %cst_4 {alignment = 8 : i32} : (!llvm.vec<8 x ptr<f64>>, vector<8xi1>, vector<8xf64>) -> vector<8xf64>
    %44 = arith.index_cast %3 : index to i64
    %45 = vector.splat %44 : vector<8xi64>
    %46 = arith.muli %45, %cst_3 : vector<8xi64>
    %47 = arith.addi %46, %cst_2 : vector<8xi64>
    %48 = memref.dim %arg0, %c0 : memref<?x4xf64>
    %49 = arith.muli %48, %c4 : index
    %50 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%49], strides: [1] : memref<?x4xf64> to memref<?xf64>
    %51 = builtin.unrealized_conversion_cast %50 : memref<?xf64> to !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %52 = builtin.unrealized_conversion_cast %c0 : index to i64
    %53 = llvm.extractvalue %51[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %54 = llvm.getelementptr %53[%52] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %55 = llvm.getelementptr %54[%47] : (!llvm.ptr<f64>, vector<8xi64>) -> !llvm.vec<8 x ptr<f64>>
    %56 = llvm.intr.masked.gather %55, %cst_5, %cst_4 {alignment = 8 : i32} : (!llvm.vec<8 x ptr<f64>>, vector<8xi1>, vector<8xf64>) -> vector<8xf64>
    %57 = llvm.fptrunc %17 : vector<8xf64> to vector<8xf32>
    %58 = arith.subf %57, %cst_8 : vector<8xf32>
    %59 = arith.mulf %58, %58 : vector<8xf32>
    %60 = arith.mulf %59, %cst_6 : vector<8xf32>
    %61 = arith.addf %cst_7, %60 : vector<8xf32>
    %62 = llvm.fptrunc %30 : vector<8xf64> to vector<8xf32>
    %63 = arith.subf %62, %cst_11 : vector<8xf32>
    %64 = arith.mulf %63, %63 : vector<8xf32>
    %65 = arith.mulf %64, %cst_9 : vector<8xf32>
    %66 = arith.addf %cst_10, %65 : vector<8xf32>
    %67 = llvm.fptrunc %43 : vector<8xf64> to vector<8xf32>
    %68 = arith.subf %67, %cst_14 : vector<8xf32>
    %69 = arith.mulf %68, %68 : vector<8xf32>
    %70 = arith.mulf %69, %cst_12 : vector<8xf32>
    %71 = arith.addf %cst_13, %70 : vector<8xf32>
    %72 = llvm.fptrunc %56 : vector<8xf64> to vector<8xf32>
    %73 = arith.subf %72, %cst_17 : vector<8xf32>
    %74 = arith.mulf %73, %73 : vector<8xf32>
    %75 = arith.mulf %74, %cst_15 : vector<8xf32>
    %76 = arith.addf %cst_16, %75 : vector<8xf32>
    %77 = arith.addf %61, %cst_18 : vector<8xf32>
    %78 = arith.addf %66, %cst_18 : vector<8xf32>
    %79 = arith.cmpf ogt, %77, %78 : vector<8xf32>
    %80 = arith.select %79, %77, %78 : vector<8xi1>, vector<8xf32>
    %81 = arith.select %79, %78, %77 : vector<8xi1>, vector<8xf32>
    %82 = arith.subf %81, %80 : vector<8xf32>
    %83 = math.exp %82 : vector<8xf32>
    %84 = math.log1p %83 : vector<8xf32>
    %85 = arith.addf %80, %84 : vector<8xf32>
    %86 = arith.addf %71, %cst_18 : vector<8xf32>
    %87 = arith.addf %76, %cst_18 : vector<8xf32>
    %88 = arith.cmpf ogt, %86, %87 : vector<8xf32>
    %89 = arith.select %88, %86, %87 : vector<8xi1>, vector<8xf32>
    %90 = arith.select %88, %87, %86 : vector<8xi1>, vector<8xf32>
    %91 = arith.subf %90, %89 : vector<8xf32>
    %92 = math.exp %91 : vector<8xf32>
    %93 = math.log1p %92 : vector<8xf32>
    %94 = arith.addf %89, %93 : vector<8xf32>
    %95 = arith.cmpf ogt, %85, %94 : vector<8xf32>
    %96 = arith.select %95, %85, %94 : vector<8xi1>, vector<8xf32>
    %97 = arith.select %95, %94, %85 : vector<8xi1>, vector<8xf32>
    %98 = arith.subf %97, %96 : vector<8xf32>
    %99 = math.exp %98 : vector<8xf32>
    %100 = math.log1p %99 : vector<8xf32>
    %101 = arith.addf %96, %100 : vector<8xf32>
    %c0_32 = arith.constant 0 : index
    %102 = memref.dim %arg1, %c0_32 : memref<?xf32>
    %cst_33 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : vector<8xi32>
    %103 = arith.index_cast %3 : index to i32
    %104 = vector.splat %103 : vector<8xi32>
    %105 = arith.addi %104, %cst_33 : vector<8xi32>
    %106 = arith.index_cast %102 : index to i32
    %107 = vector.splat %106 : vector<8xi32>
    %108 = arith.cmpi slt, %105, %107 : vector<8xi32>
    %109 = builtin.unrealized_conversion_cast %arg1 : memref<?xf32> to !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %110 = builtin.unrealized_conversion_cast %3 : index to i64
    %111 = llvm.extractvalue %109[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %112 = llvm.getelementptr %111[%110] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %113 = llvm.bitcast %112 : !llvm.ptr<f32> to !llvm.ptr<vector<8xf32>>
    llvm.intr.masked.store %101, %113, %108 {alignment = 4 : i32} : vector<8xf32>, vector<8xi1> into !llvm.ptr<vector<8xf32>>
    %114 = arith.addi %3, %c8 : index
    cf.br ^bb1(%114 : index)
  ^bb3:  // pred: ^bb1
    cf.br ^bb4(%2 : index)
  ^bb4(%115: index):  // 2 preds: ^bb3, ^bb5
    %116 = arith.cmpi slt, %115, %0 : index
    cf.cond_br %116, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %117 = memref.load %arg0[%115, %c0] : memref<?x4xf64>
    %118 = memref.load %arg0[%115, %c1] : memref<?x4xf64>
    %119 = memref.load %arg0[%115, %c2] : memref<?x4xf64>
    %120 = memref.load %arg0[%115, %c3] : memref<?x4xf64>
    %121 = llvm.fptrunc %117 : f64 to f32
    %122 = arith.subf %121, %cst_21 : f32
    %123 = arith.mulf %122, %122 : f32
    %124 = arith.mulf %123, %cst_19 : f32
    %125 = arith.addf %cst_20, %124 : f32
    %126 = llvm.fptrunc %118 : f64 to f32
    %127 = arith.subf %126, %cst_24 : f32
    %128 = arith.mulf %127, %127 : f32
    %129 = arith.mulf %128, %cst_22 : f32
    %130 = arith.addf %cst_23, %129 : f32
    %131 = llvm.fptrunc %119 : f64 to f32
    %132 = arith.subf %131, %cst_27 : f32
    %133 = arith.mulf %132, %132 : f32
    %134 = arith.mulf %133, %cst_25 : f32
    %135 = arith.addf %cst_26, %134 : f32
    %136 = llvm.fptrunc %120 : f64 to f32
    %137 = arith.subf %136, %cst_30 : f32
    %138 = arith.mulf %137, %137 : f32
    %139 = arith.mulf %138, %cst_28 : f32
    %140 = arith.addf %cst_29, %139 : f32
    %141 = arith.addf %125, %cst_31 : f32
    %142 = arith.addf %130, %cst_31 : f32
    %143 = arith.cmpf ogt, %141, %142 : f32
    %144 = arith.select %143, %141, %142 : f32
    %145 = arith.select %143, %142, %141 : f32
    %146 = arith.subf %145, %144 : f32
    %147 = math.exp %146 : f32
    %148 = math.log1p %147 : f32
    %149 = arith.addf %144, %148 : f32
    %150 = arith.addf %135, %cst_31 : f32
    %151 = arith.addf %140, %cst_31 : f32
    %152 = arith.cmpf ogt, %150, %151 : f32
    %153 = arith.select %152, %150, %151 : f32
    %154 = arith.select %152, %151, %150 : f32
    %155 = arith.subf %154, %153 : f32
    %156 = math.exp %155 : f32
    %157 = math.log1p %156 : f32
    %158 = arith.addf %153, %157 : f32
    %159 = arith.cmpf ogt, %149, %158 : f32
    %160 = arith.select %159, %149, %158 : f32
    %161 = arith.select %159, %158, %149 : f32
    %162 = arith.subf %161, %160 : f32
    %163 = math.exp %162 : f32
    %164 = math.log1p %163 : f32
    %165 = arith.addf %160, %164 : f32
    memref.store %165, %arg1[%115] : memref<?xf32>
    %166 = arith.addi %115, %c1 : index
    cf.br ^bb4(%166 : index)
  ^bb6:  // pred: ^bb4
    return
  }
  func.func @spn_vector(%arg0: memref<?x4xf64>, %arg1: memref<?xf32>) {
    call @vec_task_0(%arg0, %arg1) : (memref<?x4xf64>, memref<?xf32>) -> ()
    return
  }
}

// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

// The script is designed to make adding checks to
// a test case fast, it is *not* designed to be authoritative
// about what constitutes a good test! The CHECK should be
// minimized and named to reflect the test intent.



// CHECK-LABEL:   llvm.func @vec_task_0(
// CHECK-SAME:                          %[[VAL_0:[a-z0-9]*]]: !llvm.ptr<f64>,
// CHECK-SAME:                          %[[VAL_1:[a-z0-9]*]]: !llvm.ptr<f64>,
// CHECK-SAME:                          %[[VAL_2:[a-z0-9]*]]: i64,
// CHECK-SAME:                          %[[VAL_3:[a-z0-9]*]]: i64,
// CHECK-SAME:                          %[[VAL_4:[a-z0-9]*]]: i64,
// CHECK-SAME:                          %[[VAL_5:[a-z0-9]*]]: i64,
// CHECK-SAME:                          %[[VAL_6:[a-z0-9]*]]: i64,
// CHECK-SAME:                          %[[VAL_7:[a-z0-9]*]]: !llvm.ptr<f32>,
// CHECK-SAME:                          %[[VAL_8:[a-z0-9]*]]: !llvm.ptr<f32>,
// CHECK-SAME:                          %[[VAL_9:[a-z0-9]*]]: i64,
// CHECK-SAME:                          %[[VAL_10:[a-z0-9]*]]: i64,
// CHECK-SAME:                          %[[VAL_11:[a-z0-9]*]]: i64) {
// CHECK:           %[[VAL_12:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
// CHECK:           %[[VAL_13:.*]] = llvm.mlir.constant(dense<[0, 1, 2, 3, 4, 5, 6, 7]> : vector<8xi32>) : vector<8xi32>
// CHECK:           %[[VAL_14:.*]] = llvm.mlir.constant(dense<1.000000e+00> : vector<8xf32>) : vector<8xf32>
// CHECK:           %[[VAL_15:.*]] = llvm.mlir.constant(-1.38629436 : f32) : f32
// CHECK:           %[[VAL_16:.*]] = llvm.mlir.constant(1.400000e-01 : f32) : f32
// CHECK:           %[[VAL_17:.*]] = llvm.mlir.constant(0.467355818 : f32) : f32
// CHECK:           %[[VAL_18:.*]] = llvm.mlir.constant(-8.000000e+00 : f32) : f32
// CHECK:           %[[VAL_19:.*]] = llvm.mlir.constant(1.300000e-01 : f32) : f32
// CHECK:           %[[VAL_20:.*]] = llvm.mlir.constant(-0.22579135 : f32) : f32
// CHECK:           %[[VAL_21:.*]] = llvm.mlir.constant(-2.000000e+00 : f32) : f32
// CHECK:           %[[VAL_22:.*]] = llvm.mlir.constant(1.200000e-01 : f32) : f32
// CHECK:           %[[VAL_23:.*]] = llvm.mlir.constant(-0.631256461 : f32) : f32
// CHECK:           %[[VAL_24:.*]] = llvm.mlir.constant(-0.888888895 : f32) : f32
// CHECK:           %[[VAL_25:.*]] = llvm.mlir.constant(1.100000e-01 : f32) : f32
// CHECK:           %[[VAL_26:.*]] = llvm.mlir.constant(-0.918938517 : f32) : f32
// CHECK:           %[[VAL_27:.*]] = llvm.mlir.constant(-5.000000e-01 : f32) : f32
// CHECK:           %[[VAL_28:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK:           %[[VAL_29:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK:           %[[VAL_30:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[VAL_31:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_32:.*]] = llvm.mlir.constant(dense<-1.38629436> : vector<8xf32>) : vector<8xf32>
// CHECK:           %[[VAL_33:.*]] = llvm.mlir.constant(dense<1.400000e-01> : vector<8xf32>) : vector<8xf32>
// CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(dense<0.467355818> : vector<8xf32>) : vector<8xf32>
// CHECK:           %[[VAL_35:.*]] = llvm.mlir.constant(dense<-8.000000e+00> : vector<8xf32>) : vector<8xf32>
// CHECK:           %[[VAL_36:.*]] = llvm.mlir.constant(dense<1.300000e-01> : vector<8xf32>) : vector<8xf32>
// CHECK:           %[[VAL_37:.*]] = llvm.mlir.constant(dense<-0.22579135> : vector<8xf32>) : vector<8xf32>
// CHECK:           %[[VAL_38:.*]] = llvm.mlir.constant(dense<-2.000000e+00> : vector<8xf32>) : vector<8xf32>
// CHECK:           %[[VAL_39:.*]] = llvm.mlir.constant(dense<1.200000e-01> : vector<8xf32>) : vector<8xf32>
// CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(dense<-0.631256461> : vector<8xf32>) : vector<8xf32>
// CHECK:           %[[VAL_41:.*]] = llvm.mlir.constant(dense<-0.888888895> : vector<8xf32>) : vector<8xf32>
// CHECK:           %[[VAL_42:.*]] = llvm.mlir.constant(dense<1.100000e-01> : vector<8xf32>) : vector<8xf32>
// CHECK:           %[[VAL_43:.*]] = llvm.mlir.constant(dense<-0.918938517> : vector<8xf32>) : vector<8xf32>
// CHECK:           %[[VAL_44:.*]] = llvm.mlir.constant(dense<-5.000000e-01> : vector<8xf32>) : vector<8xf32>
// CHECK:           %[[VAL_45:.*]] = llvm.mlir.constant(4 : index) : i64
// CHECK:           %[[VAL_46:.*]] = llvm.mlir.constant(dense<true> : vector<8xi1>) : vector<8xi1>
// CHECK:           %[[VAL_47:.*]] = llvm.mlir.constant(dense<0.000000e+00> : vector<8xf64>) : vector<8xf64>
// CHECK:           %[[VAL_48:.*]] = llvm.mlir.constant(dense<4> : vector<8xi64>) : vector<8xi64>
// CHECK:           %[[VAL_49:.*]] = llvm.mlir.constant(dense<[3, 7, 11, 15, 19, 23, 27, 31]> : vector<8xi64>) : vector<8xi64>
// CHECK:           %[[VAL_50:.*]] = llvm.mlir.constant(dense<[2, 6, 10, 14, 18, 22, 26, 30]> : vector<8xi64>) : vector<8xi64>
// CHECK:           %[[VAL_51:.*]] = llvm.mlir.constant(dense<[1, 5, 9, 13, 17, 21, 25, 29]> : vector<8xi64>) : vector<8xi64>
// CHECK:           %[[VAL_52:.*]] = llvm.mlir.constant(dense<[0, 4, 8, 12, 16, 20, 24, 28]> : vector<8xi64>) : vector<8xi64>
// CHECK:           %[[VAL_53:.*]] = llvm.mlir.constant(8 : index) : i64
// CHECK:           %[[VAL_54:.*]] = llvm.urem %[[VAL_3]], %[[VAL_53]]  : i64
// CHECK:           %[[VAL_55:.*]] = llvm.sub %[[VAL_3]], %[[VAL_54]]  : i64
// CHECK:           llvm.br ^bb1(%[[VAL_31]] : i64)
// CHECK:         ^bb1(%[[VAL_56:.*]]: i64):
// CHECK:           %[[VAL_57:.*]] = llvm.icmp "slt" %[[VAL_56]], %[[VAL_55]] : i64
// CHECK:           llvm.cond_br %[[VAL_57]], ^bb2, ^bb3
// CHECK:         ^bb2:
// CHECK:           %[[VAL_58:.*]] = llvm.mlir.undef : vector<8xi64>
// CHECK:           %[[VAL_59:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_60:.*]] = llvm.insertelement %[[VAL_56]], %[[VAL_58]]{{\[}}%[[VAL_59]] : i32] : vector<8xi64>
// CHECK:           %[[VAL_61:.*]] = llvm.shufflevector %[[VAL_60]], %[[VAL_58]] [0, 0, 0, 0, 0, 0, 0, 0] : vector<8xi64>
// CHECK:           %[[VAL_62:.*]] = llvm.mul %[[VAL_61]], %[[VAL_48]]  : vector<8xi64>
// CHECK:           %[[VAL_63:.*]] = llvm.add %[[VAL_62]], %[[VAL_52]]  : vector<8xi64>
// CHECK:           %[[VAL_64:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_63]]] : (!llvm.ptr<f64>, vector<8xi64>) -> !llvm.vec<8 x ptr<f64>>
// CHECK:           %[[VAL_65:.*]] = llvm.intr.masked.gather %[[VAL_64]], %[[VAL_46]], %[[VAL_47]] {alignment = 8 : i32} : (!llvm.vec<8 x ptr<f64>>, vector<8xi1>, vector<8xf64>) -> vector<8xf64>
// CHECK:           %[[VAL_66:.*]] = llvm.mlir.undef : vector<8xi64>
// CHECK:           %[[VAL_67:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_68:.*]] = llvm.insertelement %[[VAL_56]], %[[VAL_66]]{{\[}}%[[VAL_67]] : i32] : vector<8xi64>
// CHECK:           %[[VAL_69:.*]] = llvm.shufflevector %[[VAL_68]], %[[VAL_66]] [0, 0, 0, 0, 0, 0, 0, 0] : vector<8xi64>
// CHECK:           %[[VAL_70:.*]] = llvm.mul %[[VAL_69]], %[[VAL_48]]  : vector<8xi64>
// CHECK:           %[[VAL_71:.*]] = llvm.add %[[VAL_70]], %[[VAL_51]]  : vector<8xi64>
// CHECK:           %[[VAL_72:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_71]]] : (!llvm.ptr<f64>, vector<8xi64>) -> !llvm.vec<8 x ptr<f64>>
// CHECK:           %[[VAL_73:.*]] = llvm.intr.masked.gather %[[VAL_72]], %[[VAL_46]], %[[VAL_47]] {alignment = 8 : i32} : (!llvm.vec<8 x ptr<f64>>, vector<8xi1>, vector<8xf64>) -> vector<8xf64>
// CHECK:           %[[VAL_74:.*]] = llvm.mlir.undef : vector<8xi64>
// CHECK:           %[[VAL_75:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_76:.*]] = llvm.insertelement %[[VAL_56]], %[[VAL_74]]{{\[}}%[[VAL_75]] : i32] : vector<8xi64>
// CHECK:           %[[VAL_77:.*]] = llvm.shufflevector %[[VAL_76]], %[[VAL_74]] [0, 0, 0, 0, 0, 0, 0, 0] : vector<8xi64>
// CHECK:           %[[VAL_78:.*]] = llvm.mul %[[VAL_77]], %[[VAL_48]]  : vector<8xi64>
// CHECK:           %[[VAL_79:.*]] = llvm.add %[[VAL_78]], %[[VAL_50]]  : vector<8xi64>
// CHECK:           %[[VAL_80:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_79]]] : (!llvm.ptr<f64>, vector<8xi64>) -> !llvm.vec<8 x ptr<f64>>
// CHECK:           %[[VAL_81:.*]] = llvm.intr.masked.gather %[[VAL_80]], %[[VAL_46]], %[[VAL_47]] {alignment = 8 : i32} : (!llvm.vec<8 x ptr<f64>>, vector<8xi1>, vector<8xf64>) -> vector<8xf64>
// CHECK:           %[[VAL_82:.*]] = llvm.mlir.undef : vector<8xi64>
// CHECK:           %[[VAL_83:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_84:.*]] = llvm.insertelement %[[VAL_56]], %[[VAL_82]]{{\[}}%[[VAL_83]] : i32] : vector<8xi64>
// CHECK:           %[[VAL_85:.*]] = llvm.shufflevector %[[VAL_84]], %[[VAL_82]] [0, 0, 0, 0, 0, 0, 0, 0] : vector<8xi64>
// CHECK:           %[[VAL_86:.*]] = llvm.mul %[[VAL_85]], %[[VAL_48]]  : vector<8xi64>
// CHECK:           %[[VAL_87:.*]] = llvm.add %[[VAL_86]], %[[VAL_49]]  : vector<8xi64>
// CHECK:           %[[VAL_88:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_87]]] : (!llvm.ptr<f64>, vector<8xi64>) -> !llvm.vec<8 x ptr<f64>>
// CHECK:           %[[VAL_89:.*]] = llvm.intr.masked.gather %[[VAL_88]], %[[VAL_46]], %[[VAL_47]] {alignment = 8 : i32} : (!llvm.vec<8 x ptr<f64>>, vector<8xi1>, vector<8xf64>) -> vector<8xf64>
// CHECK:           %[[VAL_90:.*]] = llvm.fptrunc %[[VAL_65]] : vector<8xf64> to vector<8xf32>
// CHECK:           %[[VAL_91:.*]] = llvm.fsub %[[VAL_90]], %[[VAL_42]]  : vector<8xf32>
// CHECK:           %[[VAL_92:.*]] = llvm.fmul %[[VAL_91]], %[[VAL_91]]  : vector<8xf32>
// CHECK:           %[[VAL_93:.*]] = llvm.fmul %[[VAL_92]], %[[VAL_44]]  : vector<8xf32>
// CHECK:           %[[VAL_94:.*]] = llvm.fadd %[[VAL_43]], %[[VAL_93]]  : vector<8xf32>
// CHECK:           %[[VAL_95:.*]] = llvm.fptrunc %[[VAL_73]] : vector<8xf64> to vector<8xf32>
// CHECK:           %[[VAL_96:.*]] = llvm.fsub %[[VAL_95]], %[[VAL_39]]  : vector<8xf32>
// CHECK:           %[[VAL_97:.*]] = llvm.fmul %[[VAL_96]], %[[VAL_96]]  : vector<8xf32>
// CHECK:           %[[VAL_98:.*]] = llvm.fmul %[[VAL_97]], %[[VAL_41]]  : vector<8xf32>
// CHECK:           %[[VAL_99:.*]] = llvm.fadd %[[VAL_40]], %[[VAL_98]]  : vector<8xf32>
// CHECK:           %[[VAL_100:.*]] = llvm.fptrunc %[[VAL_81]] : vector<8xf64> to vector<8xf32>
// CHECK:           %[[VAL_101:.*]] = llvm.fsub %[[VAL_100]], %[[VAL_36]]  : vector<8xf32>
// CHECK:           %[[VAL_102:.*]] = llvm.fmul %[[VAL_101]], %[[VAL_101]]  : vector<8xf32>
// CHECK:           %[[VAL_103:.*]] = llvm.fmul %[[VAL_102]], %[[VAL_38]]  : vector<8xf32>
// CHECK:           %[[VAL_104:.*]] = llvm.fadd %[[VAL_37]], %[[VAL_103]]  : vector<8xf32>
// CHECK:           %[[VAL_105:.*]] = llvm.fptrunc %[[VAL_89]] : vector<8xf64> to vector<8xf32>
// CHECK:           %[[VAL_106:.*]] = llvm.fsub %[[VAL_105]], %[[VAL_33]]  : vector<8xf32>
// CHECK:           %[[VAL_107:.*]] = llvm.fmul %[[VAL_106]], %[[VAL_106]]  : vector<8xf32>
// CHECK:           %[[VAL_108:.*]] = llvm.fmul %[[VAL_107]], %[[VAL_35]]  : vector<8xf32>
// CHECK:           %[[VAL_109:.*]] = llvm.fadd %[[VAL_34]], %[[VAL_108]]  : vector<8xf32>
// CHECK:           %[[VAL_110:.*]] = llvm.fadd %[[VAL_94]], %[[VAL_32]]  : vector<8xf32>
// CHECK:           %[[VAL_111:.*]] = llvm.fadd %[[VAL_99]], %[[VAL_32]]  : vector<8xf32>
// CHECK:           %[[VAL_112:.*]] = llvm.fcmp "ogt" %[[VAL_110]], %[[VAL_111]] : vector<8xf32>
// CHECK:           %[[VAL_113:.*]] = llvm.select %[[VAL_112]], %[[VAL_110]], %[[VAL_111]] : vector<8xi1>, vector<8xf32>
// CHECK:           %[[VAL_114:.*]] = llvm.select %[[VAL_112]], %[[VAL_111]], %[[VAL_110]] : vector<8xi1>, vector<8xf32>
// CHECK:           %[[VAL_115:.*]] = llvm.fsub %[[VAL_114]], %[[VAL_113]]  : vector<8xf32>
// CHECK:           %[[VAL_116:.*]] = llvm.intr.exp(%[[VAL_115]])  : (vector<8xf32>) -> vector<8xf32>
// CHECK:           %[[VAL_117:.*]] = llvm.fadd %[[VAL_14]], %[[VAL_116]]  : vector<8xf32>
// CHECK:           %[[VAL_118:.*]] = llvm.intr.log(%[[VAL_117]])  : (vector<8xf32>) -> vector<8xf32>
// CHECK:           %[[VAL_119:.*]] = llvm.fadd %[[VAL_113]], %[[VAL_118]]  : vector<8xf32>
// CHECK:           %[[VAL_120:.*]] = llvm.fadd %[[VAL_104]], %[[VAL_32]]  : vector<8xf32>
// CHECK:           %[[VAL_121:.*]] = llvm.fadd %[[VAL_109]], %[[VAL_32]]  : vector<8xf32>
// CHECK:           %[[VAL_122:.*]] = llvm.fcmp "ogt" %[[VAL_120]], %[[VAL_121]] : vector<8xf32>
// CHECK:           %[[VAL_123:.*]] = llvm.select %[[VAL_122]], %[[VAL_120]], %[[VAL_121]] : vector<8xi1>, vector<8xf32>
// CHECK:           %[[VAL_124:.*]] = llvm.select %[[VAL_122]], %[[VAL_121]], %[[VAL_120]] : vector<8xi1>, vector<8xf32>
// CHECK:           %[[VAL_125:.*]] = llvm.fsub %[[VAL_124]], %[[VAL_123]]  : vector<8xf32>
// CHECK:           %[[VAL_126:.*]] = llvm.intr.exp(%[[VAL_125]])  : (vector<8xf32>) -> vector<8xf32>
// CHECK:           %[[VAL_127:.*]] = llvm.fadd %[[VAL_14]], %[[VAL_126]]  : vector<8xf32>
// CHECK:           %[[VAL_128:.*]] = llvm.intr.log(%[[VAL_127]])  : (vector<8xf32>) -> vector<8xf32>
// CHECK:           %[[VAL_129:.*]] = llvm.fadd %[[VAL_123]], %[[VAL_128]]  : vector<8xf32>
// CHECK:           %[[VAL_130:.*]] = llvm.fcmp "ogt" %[[VAL_119]], %[[VAL_129]] : vector<8xf32>
// CHECK:           %[[VAL_131:.*]] = llvm.select %[[VAL_130]], %[[VAL_119]], %[[VAL_129]] : vector<8xi1>, vector<8xf32>
// CHECK:           %[[VAL_132:.*]] = llvm.select %[[VAL_130]], %[[VAL_129]], %[[VAL_119]] : vector<8xi1>, vector<8xf32>
// CHECK:           %[[VAL_133:.*]] = llvm.fsub %[[VAL_132]], %[[VAL_131]]  : vector<8xf32>
// CHECK:           %[[VAL_134:.*]] = llvm.intr.exp(%[[VAL_133]])  : (vector<8xf32>) -> vector<8xf32>
// CHECK:           %[[VAL_135:.*]] = llvm.fadd %[[VAL_14]], %[[VAL_134]]  : vector<8xf32>
// CHECK:           %[[VAL_136:.*]] = llvm.intr.log(%[[VAL_135]])  : (vector<8xf32>) -> vector<8xf32>
// CHECK:           %[[VAL_137:.*]] = llvm.fadd %[[VAL_131]], %[[VAL_136]]  : vector<8xf32>
// CHECK:           %[[VAL_138:.*]] = llvm.trunc %[[VAL_56]] : i64 to i32
// CHECK:           %[[VAL_139:.*]] = llvm.mlir.undef : vector<8xi32>
// CHECK:           %[[VAL_140:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_141:.*]] = llvm.insertelement %[[VAL_138]], %[[VAL_139]]{{\[}}%[[VAL_140]] : i32] : vector<8xi32>
// CHECK:           %[[VAL_142:.*]] = llvm.shufflevector %[[VAL_141]], %[[VAL_139]] [0, 0, 0, 0, 0, 0, 0, 0] : vector<8xi32>
// CHECK:           %[[VAL_143:.*]] = llvm.add %[[VAL_142]], %[[VAL_13]]  : vector<8xi32>
// CHECK:           %[[VAL_144:.*]] = llvm.trunc %[[VAL_10]] : i64 to i32
// CHECK:           %[[VAL_145:.*]] = llvm.mlir.undef : vector<8xi32>
// CHECK:           %[[VAL_146:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_147:.*]] = llvm.insertelement %[[VAL_144]], %[[VAL_145]]{{\[}}%[[VAL_146]] : i32] : vector<8xi32>
// CHECK:           %[[VAL_148:.*]] = llvm.shufflevector %[[VAL_147]], %[[VAL_145]] [0, 0, 0, 0, 0, 0, 0, 0] : vector<8xi32>
// CHECK:           %[[VAL_149:.*]] = llvm.icmp "slt" %[[VAL_143]], %[[VAL_148]] : vector<8xi32>
// CHECK:           %[[VAL_150:.*]] = llvm.getelementptr %[[VAL_8]]{{\[}}%[[VAL_56]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK:           %[[VAL_151:.*]] = llvm.bitcast %[[VAL_150]] : !llvm.ptr<f32> to !llvm.ptr<vector<8xf32>>
// CHECK:           llvm.intr.masked.store %[[VAL_137]], %[[VAL_151]], %[[VAL_149]] {alignment = 4 : i32} : vector<8xf32>, vector<8xi1> into !llvm.ptr<vector<8xf32>>
// CHECK:           %[[VAL_152:.*]] = llvm.add %[[VAL_56]], %[[VAL_53]]  : i64
// CHECK:           llvm.br ^bb1(%[[VAL_152]] : i64)
// CHECK:         ^bb3:
// CHECK:           llvm.br ^bb4(%[[VAL_55]] : i64)
// CHECK:         ^bb4(%[[VAL_153:.*]]: i64):
// CHECK:           %[[VAL_154:.*]] = llvm.icmp "slt" %[[VAL_153]], %[[VAL_3]] : i64
// CHECK:           llvm.cond_br %[[VAL_154]], ^bb5, ^bb6
// CHECK:         ^bb5:
// CHECK:           %[[VAL_155:.*]] = llvm.mul %[[VAL_153]], %[[VAL_45]]  : i64
// CHECK:           %[[VAL_156:.*]] = llvm.add %[[VAL_155]], %[[VAL_31]]  : i64
// CHECK:           %[[VAL_157:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_156]]] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
// CHECK:           %[[VAL_158:.*]] = llvm.load %[[VAL_157]] : !llvm.ptr<f64>
// CHECK:           %[[VAL_159:.*]] = llvm.mul %[[VAL_153]], %[[VAL_45]]  : i64
// CHECK:           %[[VAL_160:.*]] = llvm.add %[[VAL_159]], %[[VAL_30]]  : i64
// CHECK:           %[[VAL_161:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_160]]] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
// CHECK:           %[[VAL_162:.*]] = llvm.load %[[VAL_161]] : !llvm.ptr<f64>
// CHECK:           %[[VAL_163:.*]] = llvm.mul %[[VAL_153]], %[[VAL_45]]  : i64
// CHECK:           %[[VAL_164:.*]] = llvm.add %[[VAL_163]], %[[VAL_29]]  : i64
// CHECK:           %[[VAL_165:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_164]]] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
// CHECK:           %[[VAL_166:.*]] = llvm.load %[[VAL_165]] : !llvm.ptr<f64>
// CHECK:           %[[VAL_167:.*]] = llvm.mul %[[VAL_153]], %[[VAL_45]]  : i64
// CHECK:           %[[VAL_168:.*]] = llvm.add %[[VAL_167]], %[[VAL_28]]  : i64
// CHECK:           %[[VAL_169:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_168]]] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
// CHECK:           %[[VAL_170:.*]] = llvm.load %[[VAL_169]] : !llvm.ptr<f64>
// CHECK:           %[[VAL_171:.*]] = llvm.fptrunc %[[VAL_158]] : f64 to f32
// CHECK:           %[[VAL_172:.*]] = llvm.fsub %[[VAL_171]], %[[VAL_25]]  : f32
// CHECK:           %[[VAL_173:.*]] = llvm.fmul %[[VAL_172]], %[[VAL_172]]  : f32
// CHECK:           %[[VAL_174:.*]] = llvm.fmul %[[VAL_173]], %[[VAL_27]]  : f32
// CHECK:           %[[VAL_175:.*]] = llvm.fadd %[[VAL_26]], %[[VAL_174]]  : f32
// CHECK:           %[[VAL_176:.*]] = llvm.fptrunc %[[VAL_162]] : f64 to f32
// CHECK:           %[[VAL_177:.*]] = llvm.fsub %[[VAL_176]], %[[VAL_22]]  : f32
// CHECK:           %[[VAL_178:.*]] = llvm.fmul %[[VAL_177]], %[[VAL_177]]  : f32
// CHECK:           %[[VAL_179:.*]] = llvm.fmul %[[VAL_178]], %[[VAL_24]]  : f32
// CHECK:           %[[VAL_180:.*]] = llvm.fadd %[[VAL_23]], %[[VAL_179]]  : f32
// CHECK:           %[[VAL_181:.*]] = llvm.fptrunc %[[VAL_166]] : f64 to f32
// CHECK:           %[[VAL_182:.*]] = llvm.fsub %[[VAL_181]], %[[VAL_19]]  : f32
// CHECK:           %[[VAL_183:.*]] = llvm.fmul %[[VAL_182]], %[[VAL_182]]  : f32
// CHECK:           %[[VAL_184:.*]] = llvm.fmul %[[VAL_183]], %[[VAL_21]]  : f32
// CHECK:           %[[VAL_185:.*]] = llvm.fadd %[[VAL_20]], %[[VAL_184]]  : f32
// CHECK:           %[[VAL_186:.*]] = llvm.fptrunc %[[VAL_170]] : f64 to f32
// CHECK:           %[[VAL_187:.*]] = llvm.fsub %[[VAL_186]], %[[VAL_16]]  : f32
// CHECK:           %[[VAL_188:.*]] = llvm.fmul %[[VAL_187]], %[[VAL_187]]  : f32
// CHECK:           %[[VAL_189:.*]] = llvm.fmul %[[VAL_188]], %[[VAL_18]]  : f32
// CHECK:           %[[VAL_190:.*]] = llvm.fadd %[[VAL_17]], %[[VAL_189]]  : f32
// CHECK:           %[[VAL_191:.*]] = llvm.fadd %[[VAL_175]], %[[VAL_15]]  : f32
// CHECK:           %[[VAL_192:.*]] = llvm.fadd %[[VAL_180]], %[[VAL_15]]  : f32
// CHECK:           %[[VAL_193:.*]] = llvm.fcmp "ogt" %[[VAL_191]], %[[VAL_192]] : f32
// CHECK:           %[[VAL_194:.*]] = llvm.select %[[VAL_193]], %[[VAL_191]], %[[VAL_192]] : i1, f32
// CHECK:           %[[VAL_195:.*]] = llvm.select %[[VAL_193]], %[[VAL_192]], %[[VAL_191]] : i1, f32
// CHECK:           %[[VAL_196:.*]] = llvm.fsub %[[VAL_195]], %[[VAL_194]]  : f32
// CHECK:           %[[VAL_197:.*]] = llvm.intr.exp(%[[VAL_196]])  : (f32) -> f32
// CHECK:           %[[VAL_198:.*]] = llvm.fadd %[[VAL_12]], %[[VAL_197]]  : f32
// CHECK:           %[[VAL_199:.*]] = llvm.intr.log(%[[VAL_198]])  : (f32) -> f32
// CHECK:           %[[VAL_200:.*]] = llvm.fadd %[[VAL_194]], %[[VAL_199]]  : f32
// CHECK:           %[[VAL_201:.*]] = llvm.fadd %[[VAL_185]], %[[VAL_15]]  : f32
// CHECK:           %[[VAL_202:.*]] = llvm.fadd %[[VAL_190]], %[[VAL_15]]  : f32
// CHECK:           %[[VAL_203:.*]] = llvm.fcmp "ogt" %[[VAL_201]], %[[VAL_202]] : f32
// CHECK:           %[[VAL_204:.*]] = llvm.select %[[VAL_203]], %[[VAL_201]], %[[VAL_202]] : i1, f32
// CHECK:           %[[VAL_205:.*]] = llvm.select %[[VAL_203]], %[[VAL_202]], %[[VAL_201]] : i1, f32
// CHECK:           %[[VAL_206:.*]] = llvm.fsub %[[VAL_205]], %[[VAL_204]]  : f32
// CHECK:           %[[VAL_207:.*]] = llvm.intr.exp(%[[VAL_206]])  : (f32) -> f32
// CHECK:           %[[VAL_208:.*]] = llvm.fadd %[[VAL_12]], %[[VAL_207]]  : f32
// CHECK:           %[[VAL_209:.*]] = llvm.intr.log(%[[VAL_208]])  : (f32) -> f32
// CHECK:           %[[VAL_210:.*]] = llvm.fadd %[[VAL_204]], %[[VAL_209]]  : f32
// CHECK:           %[[VAL_211:.*]] = llvm.fcmp "ogt" %[[VAL_200]], %[[VAL_210]] : f32
// CHECK:           %[[VAL_212:.*]] = llvm.select %[[VAL_211]], %[[VAL_200]], %[[VAL_210]] : i1, f32
// CHECK:           %[[VAL_213:.*]] = llvm.select %[[VAL_211]], %[[VAL_210]], %[[VAL_200]] : i1, f32
// CHECK:           %[[VAL_214:.*]] = llvm.fsub %[[VAL_213]], %[[VAL_212]]  : f32
// CHECK:           %[[VAL_215:.*]] = llvm.intr.exp(%[[VAL_214]])  : (f32) -> f32
// CHECK:           %[[VAL_216:.*]] = llvm.fadd %[[VAL_12]], %[[VAL_215]]  : f32
// CHECK:           %[[VAL_217:.*]] = llvm.intr.log(%[[VAL_216]])  : (f32) -> f32
// CHECK:           %[[VAL_218:.*]] = llvm.fadd %[[VAL_212]], %[[VAL_217]]  : f32
// CHECK:           %[[VAL_219:.*]] = llvm.getelementptr %[[VAL_8]]{{\[}}%[[VAL_153]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK:           llvm.store %[[VAL_218]], %[[VAL_219]] : !llvm.ptr<f32>
// CHECK:           %[[VAL_220:.*]] = llvm.add %[[VAL_153]], %[[VAL_30]]  : i64
// CHECK:           llvm.br ^bb4(%[[VAL_220]] : i64)
// CHECK:         ^bb6:
// CHECK:           llvm.return
// CHECK:         }

// CHECK-LABEL:   llvm.func @spn_vector(
// CHECK-SAME:                          %[[VAL_0:[a-z0-9]*]]: !llvm.ptr<f64>,
// CHECK-SAME:                          %[[VAL_1:[a-z0-9]*]]: !llvm.ptr<f64>,
// CHECK-SAME:                          %[[VAL_2:[a-z0-9]*]]: i64,
// CHECK-SAME:                          %[[VAL_3:[a-z0-9]*]]: i64,
// CHECK-SAME:                          %[[VAL_4:[a-z0-9]*]]: i64,
// CHECK-SAME:                          %[[VAL_5:[a-z0-9]*]]: i64,
// CHECK-SAME:                          %[[VAL_6:[a-z0-9]*]]: i64,
// CHECK-SAME:                          %[[VAL_7:[a-z0-9]*]]: !llvm.ptr<f32>,
// CHECK-SAME:                          %[[VAL_8:[a-z0-9]*]]: !llvm.ptr<f32>,
// CHECK-SAME:                          %[[VAL_9:[a-z0-9]*]]: i64,
// CHECK-SAME:                          %[[VAL_10:[a-z0-9]*]]: i64,
// CHECK-SAME:                          %[[VAL_11:[a-z0-9]*]]: i64) {
// CHECK:           llvm.call @vec_task_0(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %[[VAL_6]], %[[VAL_7]], %[[VAL_8]], %[[VAL_9]], %[[VAL_10]], %[[VAL_11]]) : (!llvm.ptr<f64>, !llvm.ptr<f64>, i64, i64, i64, i64, i64, !llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64) -> ()
// CHECK:           llvm.return
// CHECK:         }
