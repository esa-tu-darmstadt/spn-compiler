include "mlir/IR/OpBase.td"

#ifndef QUERY_OP_INTERFACE
#define QUERY_OP_INTERFACE

///
/// Common interface of all SPN dialect operations
/// representing query executions.
def QueryOpInterface : OpInterface<"QueryInterface"> {

  let description = [{
    Interface for query operations to check for correct signature
    of the query function (callee).
  }];

  let methods = [
    InterfaceMethod<"Get the expected number of features in the evidence.",
                    "int", "getNumFeatures">,

    InterfaceMethod<"Get the data type of features in the evidence.",
                    "Type", "getFeatureType">,

    InterfaceMethod<"Get the data type of the result.",
                    "Type", "getResultType">
  ];

}

#endif // QUERY_OP_INTERFACE