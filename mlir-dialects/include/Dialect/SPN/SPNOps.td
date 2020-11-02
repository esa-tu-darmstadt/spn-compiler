#ifndef SPN_OPS
#define SPN_OPS

include "SPNDialect.td"
include "SPNInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/RegionKindInterface.td"

///
/// Abstract base class of n-ary arithmetic operations
///
class SPN_NAry_Op<string mnemonic> : SPN_Op<mnemonic,
  [NoSideEffect, Commutative, SameOperandsAndResultType, DeclareOpInterfaceMethods<ArithmeticOpInterface>]> {
    /// Operands: Child nodes
    /// opcount: Number of child nodes
    let arguments = (ins Variadic<AnyProbabilityType>: $operands);

    let results = (outs AnyProbabilityType);

    let verifier = [{ return ::mlir::spn::verify(*this); }];

    let hasCanonicalizer = 1;

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
def WeightedSumOp : SPN_Op<"weighted_sum",
  [NoSideEffect, SameOperandsAndResultType, DeclareOpInterfaceMethods<ArithmeticOpInterface>]> {
    let summary = "Weighted, N-ary Sum Operation";

    let description = [{
        N-ary, weighted sum operation, multiplying each
        operand with the associated, constant weight
        and adding all operands into a single result.
    }];

    /// operands: Child nodes
    /// weights: Weights associated with the cild nodes.
    /// opCount: Number of child nodes (and weights).
    let arguments = (ins Variadic<AnyProbabilityType>:$operands, F64ArrayAttr:$weights);

    let results = (outs AnyProbabilityType);

    let verifier = [{return ::mlir::spn::verify(*this); }];

    let builders = [
        OpBuilder<"llvm::ArrayRef<Value> operands, llvm::ArrayRef<double> weights">
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
///
def HistogramOp : SPN_Op<"histogram",
  [NoSideEffect, DeclareOpInterfaceMethods<LeafNodeInterface>]> {
    let summary = "histogram";

    let description =  [{
        Univariate distribution as leaf node,
        represented by a histogram.
    }];

    let arguments = (ins AnyUnsignedInteger:$index, BucketListAttr:$buckets, UI32Attr:$bucketCount);

    let results = (outs AnyProbabilityType);

    let verifier = [{ return ::mlir::spn::verify(*this); }];

    let builders = [
        OpBuilder<"Value indexVal, llvm::ArrayRef<std::tuple<int, int, double>> buckets">
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

  let arguments = (ins Variadic<AnyProbabilityType>:$retValue);

  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  let builders = [
    OpBuilder<"Value retValue">
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
  let results = (outs AnyProbabilityType);

  let builders = [
    // Build a constant with a given constant floating-point value.
    OpBuilder<"double value">
  ];

}

///
/// Single joint probability evaluation, holds the graph as region of this operation.
///
def SingleJointQuery : SPN_Op<"single_joint",
    [IsolatedFromAbove, DeclareOpInterfaceMethods<RegionKindInterface>, DeclareOpInterfaceMethods<QueryInterface>]> {

    let summary = "Single joint probability computation";

    let description = [{
        Represents a single joint probability computation by bottom-up evaluation of the
        SPN-graph.
        The single region associated with this operation represents the the SPN graph.
    }];

    let arguments = (ins UI32Attr:$numFeatures, TypeAttr:$inputType);

    let regions = (region SizedRegion<1>:$graph);

}

#endif // SPN_OPS