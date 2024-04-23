// RUN: %optcall --convert-hispn-node-to-lospn %s | FileCheck %s
// Test (re)generated by regenerate_tests.py.
module {
  "hi_spn.joint_query"() <{batchSize = 12 : ui32, errorModel = 1 : i32, featureDataType = i32, maxError = 2.000000e-02 : f64, numFeatures = 1 : ui32, queryName = "spn_kernel", supportMarginal = false}> ({
    "hi_spn.graph"() <{numFeatures = 1 : ui32}> ({
    ^bb0(%arg0: i32):
      %0 = "hi_spn.histogram"(%arg0) <{bucketCount = 2 : ui32, buckets = [#hi_spn.bucket<0 to 1 = 2.500000e-01>, #hi_spn.bucket<1 to 2 = 7.500000e-01>]}> : (i32) -> !hi_spn.probability
      "hi_spn.root"(%0) : (!hi_spn.probability) -> ()
    }) : () -> ()
  }) : () -> ()
}

// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

// The script is designed to make adding checks to
// a test case fast, it is *not* designed to be authoritative
// about what constitutes a good test! The CHECK should be
// minimized and named to reflect the test intent.



// CHECK-LABEL:   "hi_spn.joint_query"() <{batchSize = 12 : ui32, errorModel = 1 : i32, featureDataType = i32, maxError = 2.000000e-02 : f64, numFeatures = 1 : ui32, queryName = "spn_kernel", supportMarginal = false}> ({
// CHECK:           "hi_spn.graph"() <{numFeatures = 1 : ui32}> ({
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[VAL_1:.*]] = "lo_spn.histogram"(%[[VAL_0]]) <{bucketCount = 2 : ui32, buckets = [#hi_spn.bucket<0 to 1 = 2.500000e-01>, #hi_spn.bucket<1 to 2 = 7.500000e-01>], supportMarginal = false}> : (i32) -> f64
// CHECK:             %[[VAL_2:.*]] = "lo_spn.log"(%[[VAL_1]]) : (f64) -> f64
// CHECK:             "lo_spn.yield"(%[[VAL_2]]) : (f64) -> ()
// CHECK:           }) : () -> ()
// CHECK:         }) : () -> ()

