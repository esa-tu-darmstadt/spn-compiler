module  {
  "hi_spn.joint_query"() ( {
    "hi_spn.graph"() ( {
    ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64, %arg4: f64, %arg5: f64, %arg6: f64, %arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64, %arg13: f64, %arg14: f64, %arg15: f64):  // no predecessors
      %0 = "hi_spn.gaussian"(%arg0) {mean = 1.100000e-01 : f64, stddev = 1.000000e+00 : f64} : (f64) -> !hi_spn.probability
      %1 = "hi_spn.gaussian"(%arg1) {mean = 1.200000e-01 : f64, stddev = 7.500000e-01 : f64} : (f64) -> !hi_spn.probability
      %2 = "hi_spn.gaussian"(%arg2) {mean = 1.300000e-01 : f64, stddev = 5.000000e-01 : f64} : (f64) -> !hi_spn.probability
      %3 = "hi_spn.gaussian"(%arg3) {mean = 1.400000e-01 : f64, stddev = 2.500000e-01 : f64} : (f64) -> !hi_spn.probability
      %4 = "hi_spn.product"(%0, %1, %2, %3) : (!hi_spn.probability, !hi_spn.probability, !hi_spn.probability, !hi_spn.probability) -> !hi_spn.probability
      %5 = "hi_spn.gaussian"(%arg4) {mean = 2.100000e-01 : f64, stddev = 1.000000e+00 : f64} : (f64) -> !hi_spn.probability
      %6 = "hi_spn.gaussian"(%arg5) {mean = 2.200000e-01 : f64, stddev = 7.500000e-01 : f64} : (f64) -> !hi_spn.probability
      %7 = "hi_spn.gaussian"(%arg6) {mean = 2.300000e-01 : f64, stddev = 5.000000e-01 : f64} : (f64) -> !hi_spn.probability
      %8 = "hi_spn.gaussian"(%arg7) {mean = 2.400000e-01 : f64, stddev = 2.500000e-01 : f64} : (f64) -> !hi_spn.probability
      %9 = "hi_spn.product"(%5, %6, %7, %8) : (!hi_spn.probability, !hi_spn.probability, !hi_spn.probability, !hi_spn.probability) -> !hi_spn.probability
      %10 = "hi_spn.gaussian"(%arg8) {mean = 3.100000e-01 : f64, stddev = 1.000000e+00 : f64} : (f64) -> !hi_spn.probability
      %11 = "hi_spn.gaussian"(%arg9) {mean = 3.200000e-01 : f64, stddev = 7.500000e-01 : f64} : (f64) -> !hi_spn.probability
      %12 = "hi_spn.gaussian"(%arg10) {mean = 3.300000e-01 : f64, stddev = 5.000000e-01 : f64} : (f64) -> !hi_spn.probability
      %13 = "hi_spn.gaussian"(%arg11) {mean = 3.400000e-01 : f64, stddev = 2.500000e-01 : f64} : (f64) -> !hi_spn.probability
      %14 = "hi_spn.product"(%10, %11, %12, %13) : (!hi_spn.probability, !hi_spn.probability, !hi_spn.probability, !hi_spn.probability) -> !hi_spn.probability
      %15 = "hi_spn.gaussian"(%arg12) {mean = 4.100000e-01 : f64, stddev = 1.000000e+00 : f64} : (f64) -> !hi_spn.probability
      %16 = "hi_spn.gaussian"(%arg13) {mean = 4.200000e-01 : f64, stddev = 7.500000e-01 : f64} : (f64) -> !hi_spn.probability
      %17 = "hi_spn.gaussian"(%arg14) {mean = 4.300000e-01 : f64, stddev = 5.000000e-01 : f64} : (f64) -> !hi_spn.probability
      %18 = "hi_spn.gaussian"(%arg15) {mean = 4.400000e-01 : f64, stddev = 2.500000e-01 : f64} : (f64) -> !hi_spn.probability
      %19 = "hi_spn.product"(%15, %16, %17, %18) : (!hi_spn.probability, !hi_spn.probability, !hi_spn.probability, !hi_spn.probability) -> !hi_spn.probability
      %20 = "hi_spn.sum"(%4, %9, %14, %19) {weights = [2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01]} : (!hi_spn.probability, !hi_spn.probability, !hi_spn.probability, !hi_spn.probability) -> !hi_spn.probability
      "hi_spn.root"(%20) : (!hi_spn.probability) -> ()
    }) {numFeatures = 16 : ui32} : () -> ()
  }) {batchSize = 1 : ui32, errorModel = 1 : i32, inputType = f64, kernelName = "spn_vector", maxError = 2.000000e-02 : f64, numFeatures = 16 : ui32} : () -> ()
}