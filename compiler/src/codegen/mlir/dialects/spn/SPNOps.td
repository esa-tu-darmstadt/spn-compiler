#ifndef SPN_OPS
#define SPN_OPS

include "mlir/IR/OpBase.td"
include "mlir/Analysis/CallInterfaces.td"

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

  let builders = [
    OpBuilder<"Builder* b, OperationState& state, llvm::ArrayRef<Value> operands">
  ];
}

def SumOp : SPN_Op<"sum", [NoSideEffect, ]> {
  let summary = "sum";

  let description = [{
    N-ary (unweighted) sum operation adding all operands into a single result.
  }];

  let arguments = (ins Variadic<F64>:$addends, PositiveI32Attr:$opCount);

  let results = (outs F64);

  let verifier = [{ return ::verify(*this); }];

  let builders = [
    OpBuilder<"Builder* b, OperationState& state, llvm::ArrayRef<Value> operands">
  ];
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

  let builders = [
    OpBuilder<"Builder* b, OperationState& state, llvm::ArrayRef<Value> operands, llvm::ArrayRef<double> weights">
  ];
}

def LowerBoundAttr : StructFieldAttr<"lb", I32Attr>;

def UpperBoundAttr : StructFieldAttr<"ub", I32Attr>;

def HistValueAttr  : StructFieldAttr<"val", F64Attr>;

def HistBucketAttr : StructAttr<"Bucket", SPN_Dialect, [LowerBoundAttr, UpperBoundAttr, HistValueAttr]>;

def BucketListAttr : TypedArrayAttrBase<HistBucketAttr, "histogram bucket list">;

def HistogramOp : SPN_Op<"histogram", [NoSideEffect]> {
  let summary = "histogram";

  let description = [{
    Univariate distribution as leaf node, represented by a histogram.
  }];

  let arguments = (ins I32:$index, BucketListAttr:$buckets, PositiveI32Attr:$bucketCount);

  let results = (outs F64);

  let verifier = [{ return ::verify(*this); }];

  let builders = [
    OpBuilder<"Builder* b, OperationState& state, Value index, llvm::ArrayRef<std::tuple<int, int, double>> buckets">
  ];
}

def InputVarOp : SPN_Op<"input_var", [NoSideEffect]> {
  let summary = "input variable";

  let description = [{
    Reference to an input variable.
  }];

  let arguments = (ins I32Tensor:$evidence, I32Attr:$index);

  let results = (outs I32);

  let verifier = [{ return ::verify(*this); }];

  let builders = [
    OpBuilder<"Builder *b, OperationState& state, Value input, size_t index">
  ];
}

class SPNQueryOp<string mnemonic, list<OpTrait> traits = []> 
  : SPN_Op<mnemonic, !listconcat(traits, [DeclareOpInterfaceMethods<CallOpInterface>])>;

def SPNSingleQueryOp : SPNQueryOp<"query.single", []> {
  let summary = "single query";

  let description = [{
    Single SPN probabilistic (joint probability) query execution.
  }];

  let arguments = (ins FlatSymbolRefAttr:$spn, I32Tensor:$input);

  let results = (outs F64);

  let verifier = [{ return ::verify(*this); }];

  let builders = [
    OpBuilder<"Builder* b, OperationState& state, Value input, const std::string& callee">
  ];

}

def ReturnOp : SPN_Op<"return", [Terminator]> {
  let summary = "return";

  let description = [{
    Return a single query value.
  }];

  let arguments = (ins F64:$retValue);
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