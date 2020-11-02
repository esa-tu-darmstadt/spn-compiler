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
  ];

}

#endif // MLIR_DIALECT_SPN_INTERFACES