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

#endif // MLIR_DIALECT_SPN_INTERFACES