#ifndef SPN_OPS
#define SPN_OPS

include "mlir/IR/OpBase.td"
include "mlir/Analysis/CallInterfaces.td"
include "SPNOpInterfaces.td"

///
/// Dialect for SPN-specific operations.
def SPN_Dialect : Dialect {
    let name = "spn";
    let cppNamespace = "spn";
}
 
///
/// Base class of all SPN-specific operations.
class SPN_Op<string mnemonic, list<OpTrait> traits = []> : Op<SPN_Dialect, mnemonic, traits>;

///
/// Abstract base class of n-ary arithmetic operations.
class SPN_NAry_Op<string mnemonic> : SPN_Op<mnemonic, [NoSideEffect, Commutative]> {

  /// operands: Child nodes.
  /// opCount: Number of child nodes.
  let arguments = (ins Variadic<F64>:$operands, PositiveI32Attr:$opCount);

  let results = (outs F64);

  let verifier = [{ return ::verify(*this); }];

  let builders = [
    OpBuilder<"Builder* b, OperationState& state, llvm::ArrayRef<Value> operands">
  ];

  let hasCanonicalizer = 1;

}

///
/// N-ary product operation.
def ProductOp : SPN_NAry_Op<"product"> {
  let summary = "product";
  
  let description = [{
    N-ary product operation multiplies all operands into a single result.
  }];
}

///
/// Non-weighted, n-ary sum operation.
def SumOp : SPN_NAry_Op<"sum"> {
  let summary = "sum";

  let description = [{
    N-ary (unweighted) sum operation adding all operands into a single result.
  }];
}


def WeightedSumOp : SPN_Op<"weighted_sum", [NoSideEffect]> {
  let summary = "weighted sum";
  
  let description = [{
    N-ary, weighted sum operation, multiplying each operand with the associated,
    constant weight and adding all operands into a single result.
  }];

  /// operands: Child nodes.
  /// weights: Weights associated with the child nodes.
  /// opCount: Number of child nodes.
  let arguments = (ins Variadic<F64>:$operands, F64ArrayAttr:$weights, PositiveI32Attr:$opCount);

  let results = (outs F64);

  let verifier = [{ return ::verify(*this); }];

  let builders = [
    OpBuilder<"Builder* b, OperationState& state, llvm::ArrayRef<Value> operands, llvm::ArrayRef<double> weights">
  ];

  let hasCanonicalizer = 1;
}

/// Named attribute for the inclusive lower bound of a histogram bucket.
def LowerBoundAttr : StructFieldAttr<"lb", I32Attr>;

/// Named attribute for the exclusive upper bound of a histogram bucket.
def UpperBoundAttr : StructFieldAttr<"ub", I32Attr>;

/// Named attribute for the probability value of a histogram bucket.
def HistValueAttr  : StructFieldAttr<"val", F64Attr>;

/// Struct-attribute for a histogram bucket.
def HistBucketAttr : StructAttr<"Bucket", SPN_Dialect, [LowerBoundAttr, UpperBoundAttr, HistValueAttr]>;

/// List of histogram buckets as array-attribute.
def BucketListAttr : TypedArrayAttrBase<HistBucketAttr, "histogram bucket list">;

///
/// Histogram as SPN leaf node.
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

///
/// Operation holding histogram values.
def HistogramValueOp : SPN_Op<"histvalue", [NoSideEffect]> {
  let summary = "histogram values";

  let description = [{
    High-level representation of an array of probability values associated with a histogram.
    This is the 'unrolled' version of a histogram, where all buckets have been flattened into 
    a single array of values, mainly for use in non-SPN dialects (e.g. Standard).
  }];

  let arguments = (ins F64ElementsAttr:$values, I32Attr:$lb, I32Attr:$ub);

  let results = (outs F64MemRef);

  let verifier = [{ return :: verify(*this); }];

  let builders = [
    OpBuilder<"Builder* b, OperationState& state, llvm::ArrayRef<double> values, int lb, int ub">
  ];
}

///
/// SPN feature (input variable).
def InputVarOp : SPN_Op<"input_var", [NoSideEffect]> {
  let summary = "input variable";

  let description = [{
    Reference to an input variable.
  }];

  let arguments = (ins I32:$evidence, I32Attr:$index);

  let results = (outs I32);

  let verifier = [{ return ::verify(*this); }];

  let builders = [
    OpBuilder<"Builder *b, OperationState& state, Value input, size_t index">
  ];
}

///
/// Abstract base class of all operations representing SPN query execution.
class SPNQueryOp<string mnemonic, list<OpTrait> traits = []> 
  : SPN_Op<mnemonic, !listconcat(traits, [DeclareOpInterfaceMethods<CallOpInterface>,
      DeclareOpInterfaceMethods<QueryOpInterface>])>;

///
/// Single query computation of an SPN.
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

///
/// Batch computation of joint probabilities for an SPN.
def SPNJointProbBatch : SPNQueryOp<"joint.batch", []> {
  let summary = "joint probability batch";

  let description = [{
    Computation of SPN joint probability for batch of input samples.
  }];

  let arguments = (ins FlatSymbolRefAttr:$spn, I32Tensor:$input, F64Tensor:$output);

  let verifier = [{ return ::verify(*this); }];

  let builders = [
    OpBuilder<"Builder* b, OperationState& state, Value input, Value output, const std::string& callee">
  ];

}

///
/// Return operation, returning 0 or 1 probability values.
def ReturnOp : SPN_Op<"return", [Terminator]> {
  let summary = "return";

  let description = [{
    Return from SPN function, potentially returning a result.
  }];

  let arguments = (ins Variadic<F64>:$retValue);

  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  let builders = [
    OpBuilder<"Builder* b, OperationState& state, Value retValue">,
    OpBuilder<"Builder* b, OperationState& state">
  ];

}

///
/// Constant operation.
def ConstantOp : SPN_Op<"constant", [NoSideEffect]> {

  let summary = "constant";

  let description = [{
    Single floating-point value constant, e.g. for use as weight.
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins F64Attr:$value);

  // The constant operation returns a single floating-point value.
  let results = (outs F64);

  let builders = [
    // Build a constant with a given constant floating-point value.
    OpBuilder<"Builder *builder, OperationState &state, double value">
  ];

}

#endif // SPN_OPS