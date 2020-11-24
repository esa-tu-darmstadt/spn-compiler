// RUN: %drvcall %datadir/NIPS5.bin --target CPU --delete-temps no --collect-graph-stats yes --graph-stats-file %t_NIPS5.json
// RUN: cat %t_NIPS5.json | FileCheck --check-prefix=NIPS5 %s

// RUN: %drvcall %datadir/NIPS10.bin --target CPU --delete-temps no --collect-graph-stats yes --graph-stats-file %t_NIPS10.json
// RUN: cat %t_NIPS10.json | FileCheck --check-prefix=NIPS10 %s


//  Verification: Collected graph-stats match expected values for the given SPN.

// NIPS5-DAG: featureCount{{[^[:alnum:],]*}}5
// NIPS5-DAG: histogramCount{{[^[:alnum:],]*}}10
// NIPS5-DAG: innerCount{{[^[:alnum:],]*}}14
// NIPS5-DAG: leafCount{{[^[:alnum:],]*}}10
// NIPS5-DAG: productCount{{[^[:alnum:],]*}}10
// NIPS5-DAG: sumCount{{[^[:alnum:],]*}}1
// NIPS5-DAG: averageDepth{{[^[:alnum:],]*}}5.4
// NIPS5-DAG: maxDepth{{[^[:alnum:],]*}}6
// NIPS5-DAG: medianDepth{{[^[:alnum:],]*}}6.0
// NIPS5-DAG: minDepth{{[^[:alnum:],]*}}5

// NIPS10-DAG: featureCount{{[^[:alnum:],]*}}10
// NIPS10-DAG: histogramCount{{[^[:alnum:],]*}}24
// NIPS10-DAG: innerCount{{[^[:alnum:],]*}}34
// NIPS10-DAG: leafCount{{[^[:alnum:],]*}}24
// NIPS10-DAG: productCount{{[^[:alnum:],]*}}25
// NIPS10-DAG: sumCount{{[^[:alnum:],]*}}3
// NIPS10-DAG: averageDepth{{[^[:alnum:],]*}}7.166666666666667
// NIPS10-DAG: maxDepth{{[^[:alnum:],]*}}10
// NIPS10-DAG: medianDepth{{[^[:alnum:],]*}}10.0
// NIPS10-DAG: minDepth{{[^[:alnum:],]*}}6