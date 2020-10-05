// RUN: %drvcall %datadir/NIPS5.json --target CPU --delete-temps no --collect-graph-stats yes --graph-stats-file %t_NIPS5.json
// RUN: cat %t_NIPS5.json | FileCheck --check-prefix=NIPS5 %s

// RUN: %drvcall %datadir/NIPS10.json --target CPU --delete-temps no --collect-graph-stats yes --graph-stats-file %t_NIPS10.json
// RUN: cat %t_NIPS10.json | FileCheck --check-prefix=NIPS10 %s


//  Verification: Collected graph-stats match expected values for the given SPN.

// NIPS5-DAG: count_features{{[^[:alnum:],]*}}5
// NIPS5-DAG: count_nodes_histogram{{[^[:alnum:],]*}}10
// NIPS5-DAG: count_nodes_inner{{[^[:alnum:],]*}}11
// NIPS5-DAG: count_nodes_leaf{{[^[:alnum:],]*}}10
// NIPS5-DAG: count_nodes_product{{[^[:alnum:],]*}}10
// NIPS5-DAG: count_nodes_sum{{[^[:alnum:],]*}}1
// NIPS5-DAG: depth_average{{[^[:alnum:],]*}}4.4
// NIPS5-DAG: depth_max{{[^[:alnum:],]*}}5
// NIPS5-DAG: depth_median{{[^[:alnum:],]*}}4.5
// NIPS5-DAG: depth_min{{[^[:alnum:],]*}}4

// NIPS10-DAG: count_features{{[^[:alnum:],]*}}10
// NIPS10-DAG: count_nodes_histogram{{[^[:alnum:],]*}}24
// NIPS10-DAG: count_nodes_inner{{[^[:alnum:],]*}}28
// NIPS10-DAG: count_nodes_leaf{{[^[:alnum:],]*}}24
// NIPS10-DAG: count_nodes_product{{[^[:alnum:],]*}}25
// NIPS10-DAG: count_nodes_sum{{[^[:alnum:],]*}}3
// NIPS10-DAG: depth_average{{[^[:alnum:],]*}}6.166666666666667
// NIPS10-DAG: depth_max{{[^[:alnum:],]*}}9
// NIPS10-DAG: depth_median{{[^[:alnum:],]*}}6.0
// NIPS10-DAG: depth_min{{[^[:alnum:],]*}}5
