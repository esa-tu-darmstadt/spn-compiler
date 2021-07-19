// RUN: %optcall --convert-std-to-llvm %s | FileCheck %s

module  {
  func @vec_task_0(%arg0: memref<?x4xf64>, %arg1: memref<?xf32>) {
    %c8 = constant 8 : index
    %cst = constant dense<[0, 4, 8, 12, 16, 20, 24, 28]> : vector<8xi64>
    %cst_0 = constant dense<[1, 5, 9, 13, 17, 21, 25, 29]> : vector<8xi64>
    %cst_1 = constant dense<[2, 6, 10, 14, 18, 22, 26, 30]> : vector<8xi64>
    %cst_2 = constant dense<[3, 7, 11, 15, 19, 23, 27, 31]> : vector<8xi64>
    %cst_3 = constant dense<4> : vector<8xi64>
    %cst_4 = constant dense<0.000000e+00> : vector<8xf64>
    %cst_5 = constant dense<true> : vector<8xi1>
    %c4 = constant 4 : index
    %cst_6 = constant dense<-5.000000e-01> : vector<8xf32>
    %cst_7 = constant dense<-0.918938517> : vector<8xf32>
    %cst_8 = constant dense<1.100000e-01> : vector<8xf32>
    %cst_9 = constant dense<-0.888888895> : vector<8xf32>
    %cst_10 = constant dense<-0.631256461> : vector<8xf32>
    %cst_11 = constant dense<1.200000e-01> : vector<8xf32>
    %cst_12 = constant dense<-2.000000e+00> : vector<8xf32>
    %cst_13 = constant dense<-0.22579135> : vector<8xf32>
    %cst_14 = constant dense<1.300000e-01> : vector<8xf32>
    %cst_15 = constant dense<-8.000000e+00> : vector<8xf32>
    %cst_16 = constant dense<0.467355818> : vector<8xf32>
    %cst_17 = constant dense<1.400000e-01> : vector<8xf32>
    %cst_18 = constant dense<-1.38629436> : vector<8xf32>
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %c3 = constant 3 : index
    %cst_19 = constant -5.000000e-01 : f32
    %cst_20 = constant -0.918938517 : f32
    %cst_21 = constant 1.100000e-01 : f32
    %cst_22 = constant -0.888888895 : f32
    %cst_23 = constant -0.631256461 : f32
    %cst_24 = constant 1.200000e-01 : f32
    %cst_25 = constant -2.000000e+00 : f32
    %cst_26 = constant -0.22579135 : f32
    %cst_27 = constant 1.300000e-01 : f32
    %cst_28 = constant -8.000000e+00 : f32
    %cst_29 = constant 0.467355818 : f32
    %cst_30 = constant 1.400000e-01 : f32
    %cst_31 = constant -1.38629436 : f32
    %0 = memref.dim %arg0, %c0 : memref<?x4xf64>
    %1 = remi_unsigned %0, %c8 : index
    %2 = subi %0, %1 : index
    br ^bb1(%c0 : index)
  ^bb1(%3: index):  // 2 preds: ^bb0, ^bb2
    %4 = cmpi slt, %3, %2 : index
    cond_br %4, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %5 = index_cast %3 : index to i64
    %6 = splat %5 : vector<8xi64>
    %7 = muli %6, %cst_3 : vector<8xi64>
    %8 = addi %7, %cst : vector<8xi64>
    %9 = memref.dim %arg0, %c0 : memref<?x4xf64>
    %10 = muli %9, %c4 : index
    %11 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%10], strides: [1] : memref<?x4xf64> to memref<?xf64>
    %12 = llvm.mlir.cast %11 : memref<?xf64> to !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %13 = llvm.mlir.cast %c0 : index to i64
    %14 = llvm.extractvalue %12[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %15 = llvm.getelementptr %14[%13] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %16 = llvm.getelementptr %15[%8] : (!llvm.ptr<f64>, vector<8xi64>) -> !llvm.vec<8 x ptr<f64>>
    %17 = llvm.intr.masked.gather %16, %cst_5, %cst_4 {alignment = 8 : i32} : (!llvm.vec<8 x ptr<f64>>, vector<8xi1>, vector<8xf64>) -> vector<8xf64>
    %18 = index_cast %3 : index to i64
    %19 = splat %18 : vector<8xi64>
    %20 = muli %19, %cst_3 : vector<8xi64>
    %21 = addi %20, %cst_0 : vector<8xi64>
    %22 = memref.dim %arg0, %c0 : memref<?x4xf64>
    %23 = muli %22, %c4 : index
    %24 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%23], strides: [1] : memref<?x4xf64> to memref<?xf64>
    %25 = llvm.mlir.cast %24 : memref<?xf64> to !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %26 = llvm.mlir.cast %c0 : index to i64
    %27 = llvm.extractvalue %25[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %28 = llvm.getelementptr %27[%26] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %29 = llvm.getelementptr %28[%21] : (!llvm.ptr<f64>, vector<8xi64>) -> !llvm.vec<8 x ptr<f64>>
    %30 = llvm.intr.masked.gather %29, %cst_5, %cst_4 {alignment = 8 : i32} : (!llvm.vec<8 x ptr<f64>>, vector<8xi1>, vector<8xf64>) -> vector<8xf64>
    %31 = index_cast %3 : index to i64
    %32 = splat %31 : vector<8xi64>
    %33 = muli %32, %cst_3 : vector<8xi64>
    %34 = addi %33, %cst_1 : vector<8xi64>
    %35 = memref.dim %arg0, %c0 : memref<?x4xf64>
    %36 = muli %35, %c4 : index
    %37 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%36], strides: [1] : memref<?x4xf64> to memref<?xf64>
    %38 = llvm.mlir.cast %37 : memref<?xf64> to !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %39 = llvm.mlir.cast %c0 : index to i64
    %40 = llvm.extractvalue %38[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %41 = llvm.getelementptr %40[%39] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %42 = llvm.getelementptr %41[%34] : (!llvm.ptr<f64>, vector<8xi64>) -> !llvm.vec<8 x ptr<f64>>
    %43 = llvm.intr.masked.gather %42, %cst_5, %cst_4 {alignment = 8 : i32} : (!llvm.vec<8 x ptr<f64>>, vector<8xi1>, vector<8xf64>) -> vector<8xf64>
    %44 = index_cast %3 : index to i64
    %45 = splat %44 : vector<8xi64>
    %46 = muli %45, %cst_3 : vector<8xi64>
    %47 = addi %46, %cst_2 : vector<8xi64>
    %48 = memref.dim %arg0, %c0 : memref<?x4xf64>
    %49 = muli %48, %c4 : index
    %50 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%49], strides: [1] : memref<?x4xf64> to memref<?xf64>
    %51 = llvm.mlir.cast %50 : memref<?xf64> to !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %52 = llvm.mlir.cast %c0 : index to i64
    %53 = llvm.extractvalue %51[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    %54 = llvm.getelementptr %53[%52] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %55 = llvm.getelementptr %54[%47] : (!llvm.ptr<f64>, vector<8xi64>) -> !llvm.vec<8 x ptr<f64>>
    %56 = llvm.intr.masked.gather %55, %cst_5, %cst_4 {alignment = 8 : i32} : (!llvm.vec<8 x ptr<f64>>, vector<8xi1>, vector<8xf64>) -> vector<8xf64>
    %57 = fptrunc %17 : vector<8xf64> to vector<8xf32>
    %58 = subf %57, %cst_8 : vector<8xf32>
    %59 = mulf %58, %58 : vector<8xf32>
    %60 = mulf %59, %cst_6 : vector<8xf32>
    %61 = addf %cst_7, %60 : vector<8xf32>
    %62 = fptrunc %30 : vector<8xf64> to vector<8xf32>
    %63 = subf %62, %cst_11 : vector<8xf32>
    %64 = mulf %63, %63 : vector<8xf32>
    %65 = mulf %64, %cst_9 : vector<8xf32>
    %66 = addf %cst_10, %65 : vector<8xf32>
    %67 = fptrunc %43 : vector<8xf64> to vector<8xf32>
    %68 = subf %67, %cst_14 : vector<8xf32>
    %69 = mulf %68, %68 : vector<8xf32>
    %70 = mulf %69, %cst_12 : vector<8xf32>
    %71 = addf %cst_13, %70 : vector<8xf32>
    %72 = fptrunc %56 : vector<8xf64> to vector<8xf32>
    %73 = subf %72, %cst_17 : vector<8xf32>
    %74 = mulf %73, %73 : vector<8xf32>
    %75 = mulf %74, %cst_15 : vector<8xf32>
    %76 = addf %cst_16, %75 : vector<8xf32>
    %77 = addf %61, %cst_18 : vector<8xf32>
    %78 = addf %66, %cst_18 : vector<8xf32>
    %79 = cmpf ogt, %77, %78 : vector<8xf32>
    %80 = select %79, %77, %78 : vector<8xi1>, vector<8xf32>
    %81 = select %79, %78, %77 : vector<8xi1>, vector<8xf32>
    %82 = subf %81, %80 : vector<8xf32>
    %83 = math.exp %82 : vector<8xf32>
    %84 = math.log1p %83 : vector<8xf32>
    %85 = addf %80, %84 : vector<8xf32>
    %86 = addf %71, %cst_18 : vector<8xf32>
    %87 = addf %76, %cst_18 : vector<8xf32>
    %88 = cmpf ogt, %86, %87 : vector<8xf32>
    %89 = select %88, %86, %87 : vector<8xi1>, vector<8xf32>
    %90 = select %88, %87, %86 : vector<8xi1>, vector<8xf32>
    %91 = subf %90, %89 : vector<8xf32>
    %92 = math.exp %91 : vector<8xf32>
    %93 = math.log1p %92 : vector<8xf32>
    %94 = addf %89, %93 : vector<8xf32>
    %95 = cmpf ogt, %85, %94 : vector<8xf32>
    %96 = select %95, %85, %94 : vector<8xi1>, vector<8xf32>
    %97 = select %95, %94, %85 : vector<8xi1>, vector<8xf32>
    %98 = subf %97, %96 : vector<8xf32>
    %99 = math.exp %98 : vector<8xf32>
    %100 = math.log1p %99 : vector<8xf32>
    %101 = addf %96, %100 : vector<8xf32>
    %c0_32 = constant 0 : index
    %102 = memref.dim %arg1, %c0_32 : memref<?xf32>
    %cst_33 = constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : vector<8xi32>
    %103 = index_cast %3 : index to i32
    %104 = splat %103 : vector<8xi32>
    %105 = addi %104, %cst_33 : vector<8xi32>
    %106 = index_cast %102 : index to i32
    %107 = splat %106 : vector<8xi32>
    %108 = cmpi slt, %105, %107 : vector<8xi32>
    %109 = llvm.mlir.cast %arg1 : memref<?xf32> to !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %110 = llvm.mlir.cast %3 : index to i64
    %111 = llvm.extractvalue %109[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %112 = llvm.getelementptr %111[%110] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %113 = llvm.bitcast %112 : !llvm.ptr<f32> to !llvm.ptr<vector<8xf32>>
    llvm.intr.masked.store %101, %113, %108 {alignment = 4 : i32} : vector<8xf32>, vector<8xi1> into !llvm.ptr<vector<8xf32>>
    %114 = addi %3, %c8 : index
    br ^bb1(%114 : index)
  ^bb3:  // pred: ^bb1
    br ^bb4(%2 : index)
  ^bb4(%115: index):  // 2 preds: ^bb3, ^bb5
    %116 = cmpi slt, %115, %0 : index
    cond_br %116, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %117 = memref.load %arg0[%115, %c0] : memref<?x4xf64>
    %118 = memref.load %arg0[%115, %c1] : memref<?x4xf64>
    %119 = memref.load %arg0[%115, %c2] : memref<?x4xf64>
    %120 = memref.load %arg0[%115, %c3] : memref<?x4xf64>
    %121 = fptrunc %117 : f64 to f32
    %122 = subf %121, %cst_21 : f32
    %123 = mulf %122, %122 : f32
    %124 = mulf %123, %cst_19 : f32
    %125 = addf %cst_20, %124 : f32
    %126 = fptrunc %118 : f64 to f32
    %127 = subf %126, %cst_24 : f32
    %128 = mulf %127, %127 : f32
    %129 = mulf %128, %cst_22 : f32
    %130 = addf %cst_23, %129 : f32
    %131 = fptrunc %119 : f64 to f32
    %132 = subf %131, %cst_27 : f32
    %133 = mulf %132, %132 : f32
    %134 = mulf %133, %cst_25 : f32
    %135 = addf %cst_26, %134 : f32
    %136 = fptrunc %120 : f64 to f32
    %137 = subf %136, %cst_30 : f32
    %138 = mulf %137, %137 : f32
    %139 = mulf %138, %cst_28 : f32
    %140 = addf %cst_29, %139 : f32
    %141 = addf %125, %cst_31 : f32
    %142 = addf %130, %cst_31 : f32
    %143 = cmpf ogt, %141, %142 : f32
    %144 = select %143, %141, %142 : f32
    %145 = select %143, %142, %141 : f32
    %146 = subf %145, %144 : f32
    %147 = math.exp %146 : f32
    %148 = math.log1p %147 : f32
    %149 = addf %144, %148 : f32
    %150 = addf %135, %cst_31 : f32
    %151 = addf %140, %cst_31 : f32
    %152 = cmpf ogt, %150, %151 : f32
    %153 = select %152, %150, %151 : f32
    %154 = select %152, %151, %150 : f32
    %155 = subf %154, %153 : f32
    %156 = math.exp %155 : f32
    %157 = math.log1p %156 : f32
    %158 = addf %153, %157 : f32
    %159 = cmpf ogt, %149, %158 : f32
    %160 = select %159, %149, %158 : f32
    %161 = select %159, %158, %149 : f32
    %162 = subf %161, %160 : f32
    %163 = math.exp %162 : f32
    %164 = math.log1p %163 : f32
    %165 = addf %160, %164 : f32
    memref.store %165, %arg1[%115] : memref<?xf32>
    %166 = addi %115, %c1 : index
    br ^bb4(%166 : index)
  ^bb6:  // pred: ^bb4
    return
  }
  func @spn_vector(%arg0: memref<?x4xf64>, %arg1: memref<?xf32>) {
    call @vec_task_0(%arg0, %arg1) : (memref<?x4xf64>, memref<?xf32>) -> ()
    return
  }
}

// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// .. and manually adapted because the autogenerated CHECK-SAME headers matched too greedily:
// For example, consider the following, expected output line:
//     llvm.func @task_0(%arg0: !llvm.ptr<f64>, %arg1: !llvm.ptr<f64>, %arg2: i64, ...) {
// The autogenerated CHECK-SAME statement for "%arg0: !llvm.ptr<f64>" looks like this:
//     COM: CHECK-SAME:  %[[VAL_0:.*]]: !llvm.ptr<f64>,
// which does not match only "%arg0: !llvm.ptr<f64>,", but also "%arg0: !llvm.ptr<f64>, %arg1: !llvm.ptr<f64>,".
// Therefore, "%[[VAL_0:.*]]" should be replaced with "%[[VAL_0:[a-z0-9]*]]".


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
// CHECK:           %[[VAL_12:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_13:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_12]][0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_14:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_13]][1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_15:.*]] = llvm.insertvalue %[[VAL_2]], %[[VAL_14]][2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_16:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_15]][3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_17:.*]] = llvm.insertvalue %[[VAL_5]], %[[VAL_16]][4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_18:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_17]][3, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_19:.*]] = llvm.insertvalue %[[VAL_6]], %[[VAL_18]][4, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_20:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_21:.*]] = llvm.insertvalue %[[VAL_7]], %[[VAL_20]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_22:.*]] = llvm.insertvalue %[[VAL_8]], %[[VAL_21]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_23:.*]] = llvm.insertvalue %[[VAL_9]], %[[VAL_22]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_24:.*]] = llvm.insertvalue %[[VAL_10]], %[[VAL_23]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_25:.*]] = llvm.insertvalue %[[VAL_11]], %[[VAL_24]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_26:.*]] = llvm.mlir.constant(8 : index) : i64
// CHECK:           %[[VAL_27:.*]] = llvm.mlir.constant(dense<[0, 4, 8, 12, 16, 20, 24, 28]> : vector<8xi64>) : vector<8xi64>
// CHECK:           %[[VAL_28:.*]] = llvm.mlir.constant(dense<[1, 5, 9, 13, 17, 21, 25, 29]> : vector<8xi64>) : vector<8xi64>
// CHECK:           %[[VAL_29:.*]] = llvm.mlir.constant(dense<[2, 6, 10, 14, 18, 22, 26, 30]> : vector<8xi64>) : vector<8xi64>
// CHECK:           %[[VAL_30:.*]] = llvm.mlir.constant(dense<[3, 7, 11, 15, 19, 23, 27, 31]> : vector<8xi64>) : vector<8xi64>
// CHECK:           %[[VAL_31:.*]] = llvm.mlir.constant(dense<4> : vector<8xi64>) : vector<8xi64>
// CHECK:           %[[VAL_32:.*]] = llvm.mlir.constant(dense<0.000000e+00> : vector<8xf64>) : vector<8xf64>
// CHECK:           %[[VAL_33:.*]] = llvm.mlir.constant(dense<true> : vector<8xi1>) : vector<8xi1>
// CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(4 : index) : i64
// CHECK:           %[[VAL_35:.*]] = llvm.mlir.constant(dense<-5.000000e-01> : vector<8xf32>) : vector<8xf32>
// CHECK:           %[[VAL_36:.*]] = llvm.mlir.constant(dense<-0.918938517> : vector<8xf32>) : vector<8xf32>
// CHECK:           %[[VAL_37:.*]] = llvm.mlir.constant(dense<1.100000e-01> : vector<8xf32>) : vector<8xf32>
// CHECK:           %[[VAL_38:.*]] = llvm.mlir.constant(dense<-0.888888895> : vector<8xf32>) : vector<8xf32>
// CHECK:           %[[VAL_39:.*]] = llvm.mlir.constant(dense<-0.631256461> : vector<8xf32>) : vector<8xf32>
// CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(dense<1.200000e-01> : vector<8xf32>) : vector<8xf32>
// CHECK:           %[[VAL_41:.*]] = llvm.mlir.constant(dense<-2.000000e+00> : vector<8xf32>) : vector<8xf32>
// CHECK:           %[[VAL_42:.*]] = llvm.mlir.constant(dense<-0.22579135> : vector<8xf32>) : vector<8xf32>
// CHECK:           %[[VAL_43:.*]] = llvm.mlir.constant(dense<1.300000e-01> : vector<8xf32>) : vector<8xf32>
// CHECK:           %[[VAL_44:.*]] = llvm.mlir.constant(dense<-8.000000e+00> : vector<8xf32>) : vector<8xf32>
// CHECK:           %[[VAL_45:.*]] = llvm.mlir.constant(dense<0.467355818> : vector<8xf32>) : vector<8xf32>
// CHECK:           %[[VAL_46:.*]] = llvm.mlir.constant(dense<1.400000e-01> : vector<8xf32>) : vector<8xf32>
// CHECK:           %[[VAL_47:.*]] = llvm.mlir.constant(dense<-1.38629436> : vector<8xf32>) : vector<8xf32>
// CHECK:           %[[VAL_48:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_49:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[VAL_50:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK:           %[[VAL_51:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK:           %[[VAL_52:.*]] = llvm.mlir.constant(-5.000000e-01 : f32) : f32
// CHECK:           %[[VAL_53:.*]] = llvm.mlir.constant(-0.918938517 : f32) : f32
// CHECK:           %[[VAL_54:.*]] = llvm.mlir.constant(1.100000e-01 : f32) : f32
// CHECK:           %[[VAL_55:.*]] = llvm.mlir.constant(-0.888888895 : f32) : f32
// CHECK:           %[[VAL_56:.*]] = llvm.mlir.constant(-0.631256461 : f32) : f32
// CHECK:           %[[VAL_57:.*]] = llvm.mlir.constant(1.200000e-01 : f32) : f32
// CHECK:           %[[VAL_58:.*]] = llvm.mlir.constant(-2.000000e+00 : f32) : f32
// CHECK:           %[[VAL_59:.*]] = llvm.mlir.constant(-0.22579135 : f32) : f32
// CHECK:           %[[VAL_60:.*]] = llvm.mlir.constant(1.300000e-01 : f32) : f32
// CHECK:           %[[VAL_61:.*]] = llvm.mlir.constant(-8.000000e+00 : f32) : f32
// CHECK:           %[[VAL_62:.*]] = llvm.mlir.constant(0.467355818 : f32) : f32
// CHECK:           %[[VAL_63:.*]] = llvm.mlir.constant(1.400000e-01 : f32) : f32
// CHECK:           %[[VAL_64:.*]] = llvm.mlir.constant(-1.38629436 : f32) : f32
// CHECK:           %[[VAL_65:.*]] = llvm.extractvalue %[[VAL_19]][3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_66:.*]] = llvm.urem %[[VAL_65]], %[[VAL_26]]  : i64
// CHECK:           %[[VAL_67:.*]] = llvm.sub %[[VAL_65]], %[[VAL_66]]  : i64
// CHECK:           llvm.br ^bb1(%[[VAL_48]] : i64)
// CHECK:         ^bb1(%[[VAL_68:.*]]: i64):
// CHECK:           %[[VAL_69:.*]] = llvm.icmp "slt" %[[VAL_68]], %[[VAL_67]] : i64
// CHECK:           llvm.cond_br %[[VAL_69]], ^bb2, ^bb3
// CHECK:         ^bb2:
// CHECK:           %[[VAL_70:.*]] = llvm.mlir.undef : vector<8xi64>
// CHECK:           %[[VAL_71:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_72:.*]] = llvm.insertelement %[[VAL_68]], %[[VAL_70]]{{\[}}%[[VAL_71]] : i32] : vector<8xi64>
// CHECK:           %[[VAL_73:.*]] = llvm.shufflevector %[[VAL_72]], %[[VAL_70]] [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32] : vector<8xi64>, vector<8xi64>
// CHECK:           %[[VAL_74:.*]] = llvm.mul %[[VAL_73]], %[[VAL_31]]  : vector<8xi64>
// CHECK:           %[[VAL_75:.*]] = llvm.add %[[VAL_74]], %[[VAL_27]]  : vector<8xi64>
// CHECK:           %[[VAL_76:.*]] = llvm.extractvalue %[[VAL_19]][3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_77:.*]] = llvm.mul %[[VAL_76]], %[[VAL_34]]  : i64
// CHECK:           %[[VAL_78:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_79:.*]] = llvm.extractvalue %[[VAL_19]][0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_80:.*]] = llvm.extractvalue %[[VAL_19]][1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_81:.*]] = llvm.insertvalue %[[VAL_79]], %[[VAL_78]][0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_82:.*]] = llvm.insertvalue %[[VAL_80]], %[[VAL_81]][1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_83:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_84:.*]] = llvm.insertvalue %[[VAL_83]], %[[VAL_82]][2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_85:.*]] = llvm.insertvalue %[[VAL_77]], %[[VAL_84]][3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_86:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[VAL_87:.*]] = llvm.insertvalue %[[VAL_86]], %[[VAL_85]][4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_88:.*]] = llvm.extractvalue %[[VAL_87]][1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_89:.*]] = llvm.getelementptr %[[VAL_88]]{{\[}}%[[VAL_48]]] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
// CHECK:           %[[VAL_90:.*]] = llvm.getelementptr %[[VAL_89]]{{\[}}%[[VAL_75]]] : (!llvm.ptr<f64>, vector<8xi64>) -> !llvm.vec<8 x ptr<f64>>
// CHECK:           %[[VAL_91:.*]] = llvm.intr.masked.gather %[[VAL_90]], %[[VAL_33]], %[[VAL_32]] {alignment = 8 : i32} : (!llvm.vec<8 x ptr<f64>>, vector<8xi1>, vector<8xf64>) -> vector<8xf64>
// CHECK:           %[[VAL_92:.*]] = llvm.mlir.undef : vector<8xi64>
// CHECK:           %[[VAL_93:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_94:.*]] = llvm.insertelement %[[VAL_68]], %[[VAL_92]]{{\[}}%[[VAL_93]] : i32] : vector<8xi64>
// CHECK:           %[[VAL_95:.*]] = llvm.shufflevector %[[VAL_94]], %[[VAL_92]] [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32] : vector<8xi64>, vector<8xi64>
// CHECK:           %[[VAL_96:.*]] = llvm.mul %[[VAL_95]], %[[VAL_31]]  : vector<8xi64>
// CHECK:           %[[VAL_97:.*]] = llvm.add %[[VAL_96]], %[[VAL_28]]  : vector<8xi64>
// CHECK:           %[[VAL_98:.*]] = llvm.extractvalue %[[VAL_19]][3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_99:.*]] = llvm.mul %[[VAL_98]], %[[VAL_34]]  : i64
// CHECK:           %[[VAL_100:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_101:.*]] = llvm.extractvalue %[[VAL_19]][0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_102:.*]] = llvm.extractvalue %[[VAL_19]][1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_103:.*]] = llvm.insertvalue %[[VAL_101]], %[[VAL_100]][0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_104:.*]] = llvm.insertvalue %[[VAL_102]], %[[VAL_103]][1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_105:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_106:.*]] = llvm.insertvalue %[[VAL_105]], %[[VAL_104]][2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_107:.*]] = llvm.insertvalue %[[VAL_99]], %[[VAL_106]][3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_108:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[VAL_109:.*]] = llvm.insertvalue %[[VAL_108]], %[[VAL_107]][4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_110:.*]] = llvm.extractvalue %[[VAL_109]][1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_111:.*]] = llvm.getelementptr %[[VAL_110]]{{\[}}%[[VAL_48]]] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
// CHECK:           %[[VAL_112:.*]] = llvm.getelementptr %[[VAL_111]]{{\[}}%[[VAL_97]]] : (!llvm.ptr<f64>, vector<8xi64>) -> !llvm.vec<8 x ptr<f64>>
// CHECK:           %[[VAL_113:.*]] = llvm.intr.masked.gather %[[VAL_112]], %[[VAL_33]], %[[VAL_32]] {alignment = 8 : i32} : (!llvm.vec<8 x ptr<f64>>, vector<8xi1>, vector<8xf64>) -> vector<8xf64>
// CHECK:           %[[VAL_114:.*]] = llvm.mlir.undef : vector<8xi64>
// CHECK:           %[[VAL_115:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_116:.*]] = llvm.insertelement %[[VAL_68]], %[[VAL_114]]{{\[}}%[[VAL_115]] : i32] : vector<8xi64>
// CHECK:           %[[VAL_117:.*]] = llvm.shufflevector %[[VAL_116]], %[[VAL_114]] [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32] : vector<8xi64>, vector<8xi64>
// CHECK:           %[[VAL_118:.*]] = llvm.mul %[[VAL_117]], %[[VAL_31]]  : vector<8xi64>
// CHECK:           %[[VAL_119:.*]] = llvm.add %[[VAL_118]], %[[VAL_29]]  : vector<8xi64>
// CHECK:           %[[VAL_120:.*]] = llvm.extractvalue %[[VAL_19]][3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_121:.*]] = llvm.mul %[[VAL_120]], %[[VAL_34]]  : i64
// CHECK:           %[[VAL_122:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_123:.*]] = llvm.extractvalue %[[VAL_19]][0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_124:.*]] = llvm.extractvalue %[[VAL_19]][1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_125:.*]] = llvm.insertvalue %[[VAL_123]], %[[VAL_122]][0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_126:.*]] = llvm.insertvalue %[[VAL_124]], %[[VAL_125]][1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_127:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_128:.*]] = llvm.insertvalue %[[VAL_127]], %[[VAL_126]][2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_129:.*]] = llvm.insertvalue %[[VAL_121]], %[[VAL_128]][3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_130:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[VAL_131:.*]] = llvm.insertvalue %[[VAL_130]], %[[VAL_129]][4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_132:.*]] = llvm.extractvalue %[[VAL_131]][1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_133:.*]] = llvm.getelementptr %[[VAL_132]]{{\[}}%[[VAL_48]]] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
// CHECK:           %[[VAL_134:.*]] = llvm.getelementptr %[[VAL_133]]{{\[}}%[[VAL_119]]] : (!llvm.ptr<f64>, vector<8xi64>) -> !llvm.vec<8 x ptr<f64>>
// CHECK:           %[[VAL_135:.*]] = llvm.intr.masked.gather %[[VAL_134]], %[[VAL_33]], %[[VAL_32]] {alignment = 8 : i32} : (!llvm.vec<8 x ptr<f64>>, vector<8xi1>, vector<8xf64>) -> vector<8xf64>
// CHECK:           %[[VAL_136:.*]] = llvm.mlir.undef : vector<8xi64>
// CHECK:           %[[VAL_137:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_138:.*]] = llvm.insertelement %[[VAL_68]], %[[VAL_136]]{{\[}}%[[VAL_137]] : i32] : vector<8xi64>
// CHECK:           %[[VAL_139:.*]] = llvm.shufflevector %[[VAL_138]], %[[VAL_136]] [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32] : vector<8xi64>, vector<8xi64>
// CHECK:           %[[VAL_140:.*]] = llvm.mul %[[VAL_139]], %[[VAL_31]]  : vector<8xi64>
// CHECK:           %[[VAL_141:.*]] = llvm.add %[[VAL_140]], %[[VAL_30]]  : vector<8xi64>
// CHECK:           %[[VAL_142:.*]] = llvm.extractvalue %[[VAL_19]][3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_143:.*]] = llvm.mul %[[VAL_142]], %[[VAL_34]]  : i64
// CHECK:           %[[VAL_144:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_145:.*]] = llvm.extractvalue %[[VAL_19]][0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_146:.*]] = llvm.extractvalue %[[VAL_19]][1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_147:.*]] = llvm.insertvalue %[[VAL_145]], %[[VAL_144]][0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_148:.*]] = llvm.insertvalue %[[VAL_146]], %[[VAL_147]][1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_149:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_150:.*]] = llvm.insertvalue %[[VAL_149]], %[[VAL_148]][2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_151:.*]] = llvm.insertvalue %[[VAL_143]], %[[VAL_150]][3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_152:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[VAL_153:.*]] = llvm.insertvalue %[[VAL_152]], %[[VAL_151]][4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_154:.*]] = llvm.extractvalue %[[VAL_153]][1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_155:.*]] = llvm.getelementptr %[[VAL_154]]{{\[}}%[[VAL_48]]] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
// CHECK:           %[[VAL_156:.*]] = llvm.getelementptr %[[VAL_155]]{{\[}}%[[VAL_141]]] : (!llvm.ptr<f64>, vector<8xi64>) -> !llvm.vec<8 x ptr<f64>>
// CHECK:           %[[VAL_157:.*]] = llvm.intr.masked.gather %[[VAL_156]], %[[VAL_33]], %[[VAL_32]] {alignment = 8 : i32} : (!llvm.vec<8 x ptr<f64>>, vector<8xi1>, vector<8xf64>) -> vector<8xf64>
// CHECK:           %[[VAL_158:.*]] = llvm.fptrunc %[[VAL_91]] : vector<8xf64> to vector<8xf32>
// CHECK:           %[[VAL_159:.*]] = llvm.fsub %[[VAL_158]], %[[VAL_37]]  : vector<8xf32>
// CHECK:           %[[VAL_160:.*]] = llvm.fmul %[[VAL_159]], %[[VAL_159]]  : vector<8xf32>
// CHECK:           %[[VAL_161:.*]] = llvm.fmul %[[VAL_160]], %[[VAL_35]]  : vector<8xf32>
// CHECK:           %[[VAL_162:.*]] = llvm.fadd %[[VAL_36]], %[[VAL_161]]  : vector<8xf32>
// CHECK:           %[[VAL_163:.*]] = llvm.fptrunc %[[VAL_113]] : vector<8xf64> to vector<8xf32>
// CHECK:           %[[VAL_164:.*]] = llvm.fsub %[[VAL_163]], %[[VAL_40]]  : vector<8xf32>
// CHECK:           %[[VAL_165:.*]] = llvm.fmul %[[VAL_164]], %[[VAL_164]]  : vector<8xf32>
// CHECK:           %[[VAL_166:.*]] = llvm.fmul %[[VAL_165]], %[[VAL_38]]  : vector<8xf32>
// CHECK:           %[[VAL_167:.*]] = llvm.fadd %[[VAL_39]], %[[VAL_166]]  : vector<8xf32>
// CHECK:           %[[VAL_168:.*]] = llvm.fptrunc %[[VAL_135]] : vector<8xf64> to vector<8xf32>
// CHECK:           %[[VAL_169:.*]] = llvm.fsub %[[VAL_168]], %[[VAL_43]]  : vector<8xf32>
// CHECK:           %[[VAL_170:.*]] = llvm.fmul %[[VAL_169]], %[[VAL_169]]  : vector<8xf32>
// CHECK:           %[[VAL_171:.*]] = llvm.fmul %[[VAL_170]], %[[VAL_41]]  : vector<8xf32>
// CHECK:           %[[VAL_172:.*]] = llvm.fadd %[[VAL_42]], %[[VAL_171]]  : vector<8xf32>
// CHECK:           %[[VAL_173:.*]] = llvm.fptrunc %[[VAL_157]] : vector<8xf64> to vector<8xf32>
// CHECK:           %[[VAL_174:.*]] = llvm.fsub %[[VAL_173]], %[[VAL_46]]  : vector<8xf32>
// CHECK:           %[[VAL_175:.*]] = llvm.fmul %[[VAL_174]], %[[VAL_174]]  : vector<8xf32>
// CHECK:           %[[VAL_176:.*]] = llvm.fmul %[[VAL_175]], %[[VAL_44]]  : vector<8xf32>
// CHECK:           %[[VAL_177:.*]] = llvm.fadd %[[VAL_45]], %[[VAL_176]]  : vector<8xf32>
// CHECK:           %[[VAL_178:.*]] = llvm.fadd %[[VAL_162]], %[[VAL_47]]  : vector<8xf32>
// CHECK:           %[[VAL_179:.*]] = llvm.fadd %[[VAL_167]], %[[VAL_47]]  : vector<8xf32>
// CHECK:           %[[VAL_180:.*]] = llvm.fcmp "ogt" %[[VAL_178]], %[[VAL_179]] : vector<8xf32>
// CHECK:           %[[VAL_181:.*]] = llvm.select %[[VAL_180]], %[[VAL_178]], %[[VAL_179]] : vector<8xi1>, vector<8xf32>
// CHECK:           %[[VAL_182:.*]] = llvm.select %[[VAL_180]], %[[VAL_179]], %[[VAL_178]] : vector<8xi1>, vector<8xf32>
// CHECK:           %[[VAL_183:.*]] = llvm.fsub %[[VAL_182]], %[[VAL_181]]  : vector<8xf32>
// CHECK:           %[[VAL_184:.*]] = "llvm.intr.exp"(%[[VAL_183]]) : (vector<8xf32>) -> vector<8xf32>
// CHECK:           %[[VAL_185:.*]] = llvm.mlir.constant(dense<1.000000e+00> : vector<8xf32>) : vector<8xf32>
// CHECK:           %[[VAL_186:.*]] = llvm.fadd %[[VAL_185]], %[[VAL_184]]  : vector<8xf32>
// CHECK:           %[[VAL_187:.*]] = "llvm.intr.log"(%[[VAL_186]]) : (vector<8xf32>) -> vector<8xf32>
// CHECK:           %[[VAL_188:.*]] = llvm.fadd %[[VAL_181]], %[[VAL_187]]  : vector<8xf32>
// CHECK:           %[[VAL_189:.*]] = llvm.fadd %[[VAL_172]], %[[VAL_47]]  : vector<8xf32>
// CHECK:           %[[VAL_190:.*]] = llvm.fadd %[[VAL_177]], %[[VAL_47]]  : vector<8xf32>
// CHECK:           %[[VAL_191:.*]] = llvm.fcmp "ogt" %[[VAL_189]], %[[VAL_190]] : vector<8xf32>
// CHECK:           %[[VAL_192:.*]] = llvm.select %[[VAL_191]], %[[VAL_189]], %[[VAL_190]] : vector<8xi1>, vector<8xf32>
// CHECK:           %[[VAL_193:.*]] = llvm.select %[[VAL_191]], %[[VAL_190]], %[[VAL_189]] : vector<8xi1>, vector<8xf32>
// CHECK:           %[[VAL_194:.*]] = llvm.fsub %[[VAL_193]], %[[VAL_192]]  : vector<8xf32>
// CHECK:           %[[VAL_195:.*]] = "llvm.intr.exp"(%[[VAL_194]]) : (vector<8xf32>) -> vector<8xf32>
// CHECK:           %[[VAL_196:.*]] = llvm.mlir.constant(dense<1.000000e+00> : vector<8xf32>) : vector<8xf32>
// CHECK:           %[[VAL_197:.*]] = llvm.fadd %[[VAL_196]], %[[VAL_195]]  : vector<8xf32>
// CHECK:           %[[VAL_198:.*]] = "llvm.intr.log"(%[[VAL_197]]) : (vector<8xf32>) -> vector<8xf32>
// CHECK:           %[[VAL_199:.*]] = llvm.fadd %[[VAL_192]], %[[VAL_198]]  : vector<8xf32>
// CHECK:           %[[VAL_200:.*]] = llvm.fcmp "ogt" %[[VAL_188]], %[[VAL_199]] : vector<8xf32>
// CHECK:           %[[VAL_201:.*]] = llvm.select %[[VAL_200]], %[[VAL_188]], %[[VAL_199]] : vector<8xi1>, vector<8xf32>
// CHECK:           %[[VAL_202:.*]] = llvm.select %[[VAL_200]], %[[VAL_199]], %[[VAL_188]] : vector<8xi1>, vector<8xf32>
// CHECK:           %[[VAL_203:.*]] = llvm.fsub %[[VAL_202]], %[[VAL_201]]  : vector<8xf32>
// CHECK:           %[[VAL_204:.*]] = "llvm.intr.exp"(%[[VAL_203]]) : (vector<8xf32>) -> vector<8xf32>
// CHECK:           %[[VAL_205:.*]] = llvm.mlir.constant(dense<1.000000e+00> : vector<8xf32>) : vector<8xf32>
// CHECK:           %[[VAL_206:.*]] = llvm.fadd %[[VAL_205]], %[[VAL_204]]  : vector<8xf32>
// CHECK:           %[[VAL_207:.*]] = "llvm.intr.log"(%[[VAL_206]]) : (vector<8xf32>) -> vector<8xf32>
// CHECK:           %[[VAL_208:.*]] = llvm.fadd %[[VAL_201]], %[[VAL_207]]  : vector<8xf32>
// CHECK:           %[[VAL_209:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_210:.*]] = llvm.extractvalue %[[VAL_25]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_211:.*]] = llvm.mlir.constant(dense<[0, 1, 2, 3, 4, 5, 6, 7]> : vector<8xi32>) : vector<8xi32>
// CHECK:           %[[VAL_212:.*]] = llvm.trunc %[[VAL_68]] : i64 to i32
// CHECK:           %[[VAL_213:.*]] = llvm.mlir.undef : vector<8xi32>
// CHECK:           %[[VAL_214:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_215:.*]] = llvm.insertelement %[[VAL_212]], %[[VAL_213]]{{\[}}%[[VAL_214]] : i32] : vector<8xi32>
// CHECK:           %[[VAL_216:.*]] = llvm.shufflevector %[[VAL_215]], %[[VAL_213]] [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32] : vector<8xi32>, vector<8xi32>
// CHECK:           %[[VAL_217:.*]] = llvm.add %[[VAL_216]], %[[VAL_211]]  : vector<8xi32>
// CHECK:           %[[VAL_218:.*]] = llvm.trunc %[[VAL_210]] : i64 to i32
// CHECK:           %[[VAL_219:.*]] = llvm.mlir.undef : vector<8xi32>
// CHECK:           %[[VAL_220:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_221:.*]] = llvm.insertelement %[[VAL_218]], %[[VAL_219]]{{\[}}%[[VAL_220]] : i32] : vector<8xi32>
// CHECK:           %[[VAL_222:.*]] = llvm.shufflevector %[[VAL_221]], %[[VAL_219]] [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32] : vector<8xi32>, vector<8xi32>
// CHECK:           %[[VAL_223:.*]] = llvm.icmp "slt" %[[VAL_217]], %[[VAL_222]] : vector<8xi32>
// CHECK:           %[[VAL_224:.*]] = llvm.extractvalue %[[VAL_25]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_225:.*]] = llvm.getelementptr %[[VAL_224]]{{\[}}%[[VAL_68]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK:           %[[VAL_226:.*]] = llvm.bitcast %[[VAL_225]] : !llvm.ptr<f32> to !llvm.ptr<vector<8xf32>>
// CHECK:           llvm.intr.masked.store %[[VAL_208]], %[[VAL_226]], %[[VAL_223]] {alignment = 4 : i32} : vector<8xf32>, vector<8xi1> into !llvm.ptr<vector<8xf32>>
// CHECK:           %[[VAL_227:.*]] = llvm.add %[[VAL_68]], %[[VAL_26]]  : i64
// CHECK:           llvm.br ^bb1(%[[VAL_227]] : i64)
// CHECK:         ^bb3:
// CHECK:           llvm.br ^bb4(%[[VAL_67]] : i64)
// CHECK:         ^bb4(%[[VAL_228:.*]]: i64):
// CHECK:           %[[VAL_229:.*]] = llvm.icmp "slt" %[[VAL_228]], %[[VAL_65]] : i64
// CHECK:           llvm.cond_br %[[VAL_229]], ^bb5, ^bb6
// CHECK:         ^bb5:
// CHECK:           %[[VAL_230:.*]] = llvm.extractvalue %[[VAL_19]][1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_231:.*]] = llvm.mlir.constant(4 : index) : i64
// CHECK:           %[[VAL_232:.*]] = llvm.mul %[[VAL_228]], %[[VAL_231]]  : i64
// CHECK:           %[[VAL_233:.*]] = llvm.add %[[VAL_232]], %[[VAL_48]]  : i64
// CHECK:           %[[VAL_234:.*]] = llvm.getelementptr %[[VAL_230]]{{\[}}%[[VAL_233]]] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
// CHECK:           %[[VAL_235:.*]] = llvm.load %[[VAL_234]] : !llvm.ptr<f64>
// CHECK:           %[[VAL_236:.*]] = llvm.extractvalue %[[VAL_19]][1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_237:.*]] = llvm.mlir.constant(4 : index) : i64
// CHECK:           %[[VAL_238:.*]] = llvm.mul %[[VAL_228]], %[[VAL_237]]  : i64
// CHECK:           %[[VAL_239:.*]] = llvm.add %[[VAL_238]], %[[VAL_49]]  : i64
// CHECK:           %[[VAL_240:.*]] = llvm.getelementptr %[[VAL_236]]{{\[}}%[[VAL_239]]] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
// CHECK:           %[[VAL_241:.*]] = llvm.load %[[VAL_240]] : !llvm.ptr<f64>
// CHECK:           %[[VAL_242:.*]] = llvm.extractvalue %[[VAL_19]][1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_243:.*]] = llvm.mlir.constant(4 : index) : i64
// CHECK:           %[[VAL_244:.*]] = llvm.mul %[[VAL_228]], %[[VAL_243]]  : i64
// CHECK:           %[[VAL_245:.*]] = llvm.add %[[VAL_244]], %[[VAL_50]]  : i64
// CHECK:           %[[VAL_246:.*]] = llvm.getelementptr %[[VAL_242]]{{\[}}%[[VAL_245]]] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
// CHECK:           %[[VAL_247:.*]] = llvm.load %[[VAL_246]] : !llvm.ptr<f64>
// CHECK:           %[[VAL_248:.*]] = llvm.extractvalue %[[VAL_19]][1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_249:.*]] = llvm.mlir.constant(4 : index) : i64
// CHECK:           %[[VAL_250:.*]] = llvm.mul %[[VAL_228]], %[[VAL_249]]  : i64
// CHECK:           %[[VAL_251:.*]] = llvm.add %[[VAL_250]], %[[VAL_51]]  : i64
// CHECK:           %[[VAL_252:.*]] = llvm.getelementptr %[[VAL_248]]{{\[}}%[[VAL_251]]] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
// CHECK:           %[[VAL_253:.*]] = llvm.load %[[VAL_252]] : !llvm.ptr<f64>
// CHECK:           %[[VAL_254:.*]] = llvm.fptrunc %[[VAL_235]] : f64 to f32
// CHECK:           %[[VAL_255:.*]] = llvm.fsub %[[VAL_254]], %[[VAL_54]]  : f32
// CHECK:           %[[VAL_256:.*]] = llvm.fmul %[[VAL_255]], %[[VAL_255]]  : f32
// CHECK:           %[[VAL_257:.*]] = llvm.fmul %[[VAL_256]], %[[VAL_52]]  : f32
// CHECK:           %[[VAL_258:.*]] = llvm.fadd %[[VAL_53]], %[[VAL_257]]  : f32
// CHECK:           %[[VAL_259:.*]] = llvm.fptrunc %[[VAL_241]] : f64 to f32
// CHECK:           %[[VAL_260:.*]] = llvm.fsub %[[VAL_259]], %[[VAL_57]]  : f32
// CHECK:           %[[VAL_261:.*]] = llvm.fmul %[[VAL_260]], %[[VAL_260]]  : f32
// CHECK:           %[[VAL_262:.*]] = llvm.fmul %[[VAL_261]], %[[VAL_55]]  : f32
// CHECK:           %[[VAL_263:.*]] = llvm.fadd %[[VAL_56]], %[[VAL_262]]  : f32
// CHECK:           %[[VAL_264:.*]] = llvm.fptrunc %[[VAL_247]] : f64 to f32
// CHECK:           %[[VAL_265:.*]] = llvm.fsub %[[VAL_264]], %[[VAL_60]]  : f32
// CHECK:           %[[VAL_266:.*]] = llvm.fmul %[[VAL_265]], %[[VAL_265]]  : f32
// CHECK:           %[[VAL_267:.*]] = llvm.fmul %[[VAL_266]], %[[VAL_58]]  : f32
// CHECK:           %[[VAL_268:.*]] = llvm.fadd %[[VAL_59]], %[[VAL_267]]  : f32
// CHECK:           %[[VAL_269:.*]] = llvm.fptrunc %[[VAL_253]] : f64 to f32
// CHECK:           %[[VAL_270:.*]] = llvm.fsub %[[VAL_269]], %[[VAL_63]]  : f32
// CHECK:           %[[VAL_271:.*]] = llvm.fmul %[[VAL_270]], %[[VAL_270]]  : f32
// CHECK:           %[[VAL_272:.*]] = llvm.fmul %[[VAL_271]], %[[VAL_61]]  : f32
// CHECK:           %[[VAL_273:.*]] = llvm.fadd %[[VAL_62]], %[[VAL_272]]  : f32
// CHECK:           %[[VAL_274:.*]] = llvm.fadd %[[VAL_258]], %[[VAL_64]]  : f32
// CHECK:           %[[VAL_275:.*]] = llvm.fadd %[[VAL_263]], %[[VAL_64]]  : f32
// CHECK:           %[[VAL_276:.*]] = llvm.fcmp "ogt" %[[VAL_274]], %[[VAL_275]] : f32
// CHECK:           %[[VAL_277:.*]] = llvm.select %[[VAL_276]], %[[VAL_274]], %[[VAL_275]] : i1, f32
// CHECK:           %[[VAL_278:.*]] = llvm.select %[[VAL_276]], %[[VAL_275]], %[[VAL_274]] : i1, f32
// CHECK:           %[[VAL_279:.*]] = llvm.fsub %[[VAL_278]], %[[VAL_277]]  : f32
// CHECK:           %[[VAL_280:.*]] = "llvm.intr.exp"(%[[VAL_279]]) : (f32) -> f32
// CHECK:           %[[VAL_281:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
// CHECK:           %[[VAL_282:.*]] = llvm.fadd %[[VAL_281]], %[[VAL_280]]  : f32
// CHECK:           %[[VAL_283:.*]] = "llvm.intr.log"(%[[VAL_282]]) : (f32) -> f32
// CHECK:           %[[VAL_284:.*]] = llvm.fadd %[[VAL_277]], %[[VAL_283]]  : f32
// CHECK:           %[[VAL_285:.*]] = llvm.fadd %[[VAL_268]], %[[VAL_64]]  : f32
// CHECK:           %[[VAL_286:.*]] = llvm.fadd %[[VAL_273]], %[[VAL_64]]  : f32
// CHECK:           %[[VAL_287:.*]] = llvm.fcmp "ogt" %[[VAL_285]], %[[VAL_286]] : f32
// CHECK:           %[[VAL_288:.*]] = llvm.select %[[VAL_287]], %[[VAL_285]], %[[VAL_286]] : i1, f32
// CHECK:           %[[VAL_289:.*]] = llvm.select %[[VAL_287]], %[[VAL_286]], %[[VAL_285]] : i1, f32
// CHECK:           %[[VAL_290:.*]] = llvm.fsub %[[VAL_289]], %[[VAL_288]]  : f32
// CHECK:           %[[VAL_291:.*]] = "llvm.intr.exp"(%[[VAL_290]]) : (f32) -> f32
// CHECK:           %[[VAL_292:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
// CHECK:           %[[VAL_293:.*]] = llvm.fadd %[[VAL_292]], %[[VAL_291]]  : f32
// CHECK:           %[[VAL_294:.*]] = "llvm.intr.log"(%[[VAL_293]]) : (f32) -> f32
// CHECK:           %[[VAL_295:.*]] = llvm.fadd %[[VAL_288]], %[[VAL_294]]  : f32
// CHECK:           %[[VAL_296:.*]] = llvm.fcmp "ogt" %[[VAL_284]], %[[VAL_295]] : f32
// CHECK:           %[[VAL_297:.*]] = llvm.select %[[VAL_296]], %[[VAL_284]], %[[VAL_295]] : i1, f32
// CHECK:           %[[VAL_298:.*]] = llvm.select %[[VAL_296]], %[[VAL_295]], %[[VAL_284]] : i1, f32
// CHECK:           %[[VAL_299:.*]] = llvm.fsub %[[VAL_298]], %[[VAL_297]]  : f32
// CHECK:           %[[VAL_300:.*]] = "llvm.intr.exp"(%[[VAL_299]]) : (f32) -> f32
// CHECK:           %[[VAL_301:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
// CHECK:           %[[VAL_302:.*]] = llvm.fadd %[[VAL_301]], %[[VAL_300]]  : f32
// CHECK:           %[[VAL_303:.*]] = "llvm.intr.log"(%[[VAL_302]]) : (f32) -> f32
// CHECK:           %[[VAL_304:.*]] = llvm.fadd %[[VAL_297]], %[[VAL_303]]  : f32
// CHECK:           %[[VAL_305:.*]] = llvm.extractvalue %[[VAL_25]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_306:.*]] = llvm.getelementptr %[[VAL_305]]{{\[}}%[[VAL_228]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK:           llvm.store %[[VAL_304]], %[[VAL_306]] : !llvm.ptr<f32>
// CHECK:           %[[VAL_307:.*]] = llvm.add %[[VAL_228]], %[[VAL_49]]  : i64
// CHECK:           llvm.br ^bb4(%[[VAL_307]] : i64)
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
// CHECK:           %[[VAL_12:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_13:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_12]][0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_14:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_13]][1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_15:.*]] = llvm.insertvalue %[[VAL_2]], %[[VAL_14]][2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_16:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_15]][3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_17:.*]] = llvm.insertvalue %[[VAL_5]], %[[VAL_16]][4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_18:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_17]][3, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_19:.*]] = llvm.insertvalue %[[VAL_6]], %[[VAL_18]][4, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_20:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_21:.*]] = llvm.insertvalue %[[VAL_7]], %[[VAL_20]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_22:.*]] = llvm.insertvalue %[[VAL_8]], %[[VAL_21]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_23:.*]] = llvm.insertvalue %[[VAL_9]], %[[VAL_22]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_24:.*]] = llvm.insertvalue %[[VAL_10]], %[[VAL_23]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_25:.*]] = llvm.insertvalue %[[VAL_11]], %[[VAL_24]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_26:.*]] = llvm.extractvalue %[[VAL_19]][0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_27:.*]] = llvm.extractvalue %[[VAL_19]][1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_28:.*]] = llvm.extractvalue %[[VAL_19]][2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_29:.*]] = llvm.extractvalue %[[VAL_19]][3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_30:.*]] = llvm.extractvalue %[[VAL_19]][3, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_31:.*]] = llvm.extractvalue %[[VAL_19]][4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_32:.*]] = llvm.extractvalue %[[VAL_19]][4, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[VAL_33:.*]] = llvm.extractvalue %[[VAL_25]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_34:.*]] = llvm.extractvalue %[[VAL_25]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_35:.*]] = llvm.extractvalue %[[VAL_25]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_36:.*]] = llvm.extractvalue %[[VAL_25]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[VAL_37:.*]] = llvm.extractvalue %[[VAL_25]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           llvm.call @vec_task_0(%[[VAL_26]], %[[VAL_27]], %[[VAL_28]], %[[VAL_29]], %[[VAL_30]], %[[VAL_31]], %[[VAL_32]], %[[VAL_33]], %[[VAL_34]], %[[VAL_35]], %[[VAL_36]], %[[VAL_37]]) : (!llvm.ptr<f64>, !llvm.ptr<f64>, i64, i64, i64, i64, i64, !llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64) -> ()
// CHECK:           llvm.return
// CHECK:         }
