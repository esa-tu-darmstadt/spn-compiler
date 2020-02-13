#ifndef SPN_OPS
#define SPN_OPS

include "mlir/IR/OpBase.td"

class ArrayAttrElemCount<string name> :
    StrFunc<"$" # name # ".cast<ArrayAttr>().size()">;

def SPN_Dialect : Dialect {
    let name = "spn";
    let cppNamespace = "spn";
}

class SPN_Op<string mnemonic, list<OpTrait> traits = []> : Op<SPN_Dialect, mnemonic, traits>;

def ProductOp : SPN_Op<"product", [NoSideEffect]> {
  let summary = "product";
  
  let description = [{
    N-ary product operation multiplies all operands into a single result.
  }];

  let arguments = (ins Variadic<F64>:$multiplicands, PositiveI32Attr:$opCount);

  let results = (outs F64);

  let verifier = [{ return ::verify(*this); }];
}

def SumOp : SPN_Op<"sum", [NoSideEffect, ]> {
  let summary = "sum";

  let description = [{
    N-ary (unweighted) sum operation adding all operands into a single result.
  }];

  let arguments = (ins Variadic<F64>:$addends, PositiveI32Attr:$opCount);

  let results = (outs F64);

  let verifier = [{ return ::verify(*this); }];
}

def WeightedSumOp : SPN_Op<"weighted_sum", [NoSideEffect]> {
  let summary = "weighted sum";
  
  let description = [{
    N-ary, weighted sum operation, multiplying each operand with the associated,
    constant weight and adding all operands into a single result.
  }];

  let arguments = (ins Variadic<F64>:$operands, F64ArrayAttr:$weights, PositiveI32Attr:$opCount);

  let results = (outs F64);

  let verifier = [{ return ::verify(*this); }];
}


def ConstantOp : SPN_Op<"constant", [NoSideEffect]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "toy.constant"()
         { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
        : () -> tensor<2x3xf64>
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins F64ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs F64Tensor);

  // Add custom build methods for the constant operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantOp>(...)`.
  let builders = [
    // Build a constant with a given constant tensor value.
    OpBuilder<"Builder *builder, OperationState &state, "
              "DenseElementsAttr value", [{
      build(builder, state, value.getType(), value);
    }]>,

    // Build a constant with a given constant floating-point value.
    OpBuilder<"Builder *builder, OperationState &state, double value">
  ];

  // Invoke a static verify method to verify this constant operation.
  let verifier = [{ return ::verify(*this); }];
}

#endif // SPN_OPS