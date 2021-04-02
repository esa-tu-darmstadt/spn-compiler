module  {
  "hi_spn.joint_query"() ( {
    "hi_spn.graph"() ( {
    ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64, %arg4: f64, %arg5: f64, %arg6: f64, %arg7: f64, %arg8: f64, %arg9: f64, %arg10: f64, %arg11: f64, %arg12: f64, %arg13: f64, %arg14: f64, %arg15: f64):  // no predecessors
      %4 = "lo_spn.mul"(%arg0, %arg1) : (f64, f64) -> f64
      %5 = "lo_spn.mul"(%arg2, %arg3) : (f64, f64) -> f64
      %6 = "lo_spn.mul"(%4, %5) : (f64, f64) -> f64
	  
      %11 = "lo_spn.mul"(%arg4, %arg5) : (f64, f64) -> f64
      %12 = "lo_spn.mul"(%arg6, %arg7) : (f64, f64) -> f64
      %13 = "lo_spn.mul"(%11, %12) : (f64, f64) -> f64
	  
      %14 = "lo_spn.mul"(%arg8, %arg9) : (f64, f64) -> f64
      %15 = "lo_spn.mul"(%arg10, %arg11) : (f64, f64) -> f64
      %16 = "lo_spn.mul"(%14, %15) : (f64, f64) -> f64
	  
      %19 = "lo_spn.mul"(%arg12, %arg13) : (f64, f64) -> f64
      %20 = "lo_spn.mul"(%arg14, %arg15) : (f64, f64) -> f64
      %21 = "lo_spn.mul"(%19, %20) : (f64, f64) -> f64
	  
	  
      %22 = "lo_spn.constant"() {type = f64, value = 2.500000e-01 : f64} : () -> f64
      %23 = "lo_spn.mul"(%6, %22) : (f64, f64) -> f64
	  
	  
      %24 = "lo_spn.constant"() {type = f64, value = 2.500000e-01 : f64} : () -> f64
      %25 = "lo_spn.mul"(%13, %24) : (f64, f64) -> f64
	  
      %26 = "lo_spn.add"(%23, %25) : (f64, f64) -> f64
	  
	  
      %27 = "lo_spn.constant"() {type = f64, value = 2.500000e-01 : f64} : () -> f64
      %28 = "lo_spn.mul"(%16, %27) : (f64, f64) -> f64
	  
	  
      %29 = "lo_spn.constant"() {type = f64, value = 2.500000e-01 : f64} : () -> f64
      %30 = "lo_spn.mul"(%21, %29) : (f64, f64) -> f64
	  
	  
      %31 = "lo_spn.add"(%28, %30) : (f64, f64) -> f64
	  
	  
      %32 = "lo_spn.add"(%26, %31) : (f64, f64) -> f64
	  
	  
      %33 = "lo_spn.log"(%32) : (f64) -> f64
      "lo_spn.yield"(%33) : (f64) -> ()
    }) {numFeatures = 16 : ui32} : () -> ()
  }) {batchSize = 1 : ui32, errorModel = 1 : i32, inputType = f64, kernelName = "spn_vector", maxError = 2.000000e-02 : f64, numFeatures = 16 : ui32} : () -> ()
}