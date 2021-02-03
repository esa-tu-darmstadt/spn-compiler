#ifndef MLIR_DIALECT_SPN_INTERFACES
#define MLIR_DIALECT_SPN_INTERFACES

include "mlir/IR/OpBase.td"

def ArithmeticOpInterface : OpInterface<"ArithmeticOpInterface"> {
  let description = [{
    Interface provided by all arithmetic operations in the SPN MLIR dialect.
  }];

  let methods = [
    InterfaceMethod<"Get the number of operands.", "unsigned", "getNumOperands">,
  ];
}

def LeafNodeInterface : OpInterface<"LeafNodeInterface"> {

  let description = [{
    Interface common to all SPN leaf nodes.
  }];

  let methods = [
    InterfaceMethod<"Get the feature index.", "unsigned", "getFeatureIndex">,
  ];

}

def QueryInterface : OpInterface<"QueryInterface"> {

  let description = [{
    Interface common to all SPN queries.
  }];

  let methods = [
    InterfaceMethod<"Get the graph root nodes.", "std::vector<Operation*>", "getRootNodes">,
    InterfaceMethod<"Get the number of features.", "unsigned int", "getNumFeatures">,
    InterfaceMethod<"Get the data type of the features", "mlir::Type", "getFeatureDataType">,
    InterfaceMethod<"Get the data type used for computation", "mlir::Type", "getComputationDataType">,
    InterfaceMethod<"Get the batch size to optimize for.", "unsigned int", "getBatchSize">,
    InterfaceMethod<"Get the error model for arithmetic error analysis.", "mlir::spn::error_model", "getErrorModel">,
    InterfaceMethod<"Get the maximum allowed arithmetic error.", "double", "getMaxError">,
    InterfaceMethod<"Get the query name.", "llvm::StringRef", "getQueryName">,
  ];

}

#endif // MLIR_DIALECT_SPN_INTERFACES