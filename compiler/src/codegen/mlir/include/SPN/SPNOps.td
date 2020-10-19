#ifndef SPN_OPS
#define SPN_OPS

include "SPNDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

///
/// Abstract base class of n-ary arithmetic operations
///
class SPN_NAry_Op<string mnemonic> : SPN_Op<mnemonic, [NoSideEffect, Commutative, SameOperandsAndResultType]> {
    /// Operands: Child nodes
    /// opcount: Number of child nodes
    let arguments = (ins Variadic<F64>: $operands, UI32Attr: $opCount);

    let results = (outs F64);

    let verifier = [{ return ::mlir::spn::verify(*this); }];

    let builders = [
        OpBuilder<"llvm::ArrayRef<Value> operands">
    ];

}

///
/// N-ary product operation.
///
def ProductOp : SPN_NAry_Op<"product"> {
    let summary = "N-ary Product (multiplication) operation";
}

///
/// (Non-weighted) N-ary sum operation.
///
def SumOp : SPN_NAry_Op<"sum"> {
    let summary = "N-ary Sum operation";
}

///
/// Weighted N-ary sum operation.
///
def WeightedSumOp : SPN_Op<"weighted_sum", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "Weighted, N-ary Sum Operation";

    let description = [{
        N-ary, weighted sum operation, multiplying each
        operand with the associated, constant weight
        and adding all operands into a single result.
    }];

    /// operands: Child nodes
    /// weights: Weights associated with the cild nodes.
    /// opCount: Number of child nodes (and weights).
    let arguments = (ins Variadic<F64>:$operands, F64ArrayAttr:$weights, UI32Attr:$opCount);

    let results = (outs F64);

    let verifier = [{return ::mlir::spn::verify(*this); }];

    let builders = [
        OpBuilder<"llvm::ArrayRef<Value> operands, llvm::ArrayRef<double> weights">
    ];

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
///
def HistogramOp : SPN_Op<"histogram", [NoSideEffect]> {
    let summary = "histogram";

    let description =  [{
        Univariate distribution as leaf node,
        represented by a histogram.
    }];

    let arguments = (ins I32:$index, BucketListAttr:$buckets, UI32Attr:$bucketCount);

    let results = (outs F64);

    let verifier = [{ return ::mlir::spn::verify(*this); }];

    let builders = [
        OpBuilder<"llvm::ArrayRef<std::tuple<int, int, double>> buckets">
    ];
}

///
/// Return operation, returning 0 or 1 probability values.
///
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
///
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