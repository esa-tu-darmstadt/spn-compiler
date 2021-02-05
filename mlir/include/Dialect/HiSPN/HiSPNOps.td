#ifndef HiSPN_OPS
#define HiSPN_OPS

include "HiSPNDialect.td"
include "HiSPNInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/RegionKindInterface.td"

///
/// Attributes representing different kinds of arithmetic errors:
///
def AbsoluteErrorAttr : I32EnumAttrCase<"absolute_error", 1>;
def RelativeErrorAttr : I32EnumAttrCase<"relative_error", 2>;
def ErrorModelAttr : I32EnumAttr<"error_model", "Arithmetic error kind", [AbsoluteErrorAttr, RelativeErrorAttr]> {
    let cppNamespace = "mlir::spn::high";
}

///
/// Single joint probability evaluation, holds the graph as region of this operation.
///
def JointQuery : HiSPN_Op<"joint_query",
    [IsolatedFromAbove, DeclareOpInterfaceMethods<QueryInterface>,
    SingleBlockImplicitTerminator<"Graph">]> {

    let summary = "Joint probability computation";

    let description = [{
        Represents a joint probability computation by bottom-up evaluation of the
        SPN-graph.
        The single region associated with this operation holds the the SPN graph.
    }];

    let arguments = (ins UI32Attr:$numFeatures,
                    TypeAttr:$inputType,
                    SymbolNameAttr:$kernelName,
                    UI32Attr:$batchSize,
                    ErrorModelAttr:$errorModel,
                    F64Attr:$maxError
                    );

    let regions = (region SizedRegion<1>:$graph);

}

///
/// Operation representing the actual directed acyclic graph making up the Sum-Product Network.
///
def Graph : HiSPN_Op<"graph",
    [DeclareOpInterfaceMethods<RegionKindInterface>, IsolatedFromAbove,
    // TODO Use ParentOneOf<list<string> opList> as soon as we have multiple queries
    HasParent<"JointQuery">,
    SingleBlockImplicitTerminator<"RootNode">,
    // The graph must be a terminator himself to live in the query's region.
    Terminator]> {

    let summary = "Sum-Product Network graph";

    let description = [{
        Representation of a Sum-Product Network as a directed acyclic graph (DAG).
        The single region holds the actual nodes of the SPN (sum, product, leaves).
    }];

    let arguments = (ins UI32Attr:$numFeatures);

    let regions = (region SizedRegion<1>:$graph);

}

///
/// Abstract base class of all operations that can be part of the SPN DAG.
///
class HiSPNGraphOp<string mnemonic, list<OpTrait> traits = []> : HiSPN_Op<mnemonic,
    !listconcat(traits, [HasParent<"Graph">])>;

///
/// Terminates the DAG, represents the root node of the SPN.
///
def RootNode : HiSPNGraphOp<"root", [Terminator]> {

    let summary = "Root node of the SPN graph";

    let description = [{
        Single root node of the directed acyclic graph representing the SPN.
    }];

    let arguments = (ins HiSPNProbabilityType:$root);

}

///
/// N-ary Product node inside the SPN DAG.
///
def ProductNode : HiSPNGraphOp<"product", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "N-Ary SPN Product node";

    let arguments = (ins Variadic<HiSPNProbabilityType>:$operands);

    let results = (outs HiSPNProbabilityType);

    let verifier = [{return ::mlir::spn::high::verify(*this); }];

}

///
/// N-ary Sum node inside the SPN DAG.
///
def SumNode : HiSPNGraphOp<"sum", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "Weighted N-ary SPN Sum node";

    let description = [{
        N-ary, weighted sum operation, multiplying each
        operand with the associated, constant weight
        and adding all operands into a single result.
    }];

    let arguments = (ins Variadic<HiSPNProbabilityType>:$operands, F64ArrayAttr:$weights);

    let results = (outs HiSPNProbabilityType);

    let verifier = [{return ::mlir::spn::high::verify(*this); }];

    let builders = [
        OpBuilderDAG<(ins "llvm::ArrayRef<Value>":$operands, "llvm::ArrayRef<double>":$weights)>
    ];

}

/// Named attribute for the inclusive lower bound of a histogram bucket.
def LowerBoundAttr : StructFieldAttr<"lb", I32Attr>;

/// Named attribute for the exclusive upper bound of a histogram bucket.
def UpperBoundAttr : StructFieldAttr<"ub", I32Attr>;

/// Named attribute for the probability value of a histogram bucket.
def HistValueAttr  : StructFieldAttr<"val", F64Attr>;

/// Struct-attribute for a histogram bucket.
def HistBucketAttr : StructAttr<"Bucket", HiSPN_Dialect, [LowerBoundAttr, UpperBoundAttr, HistValueAttr]>;

/// List of histogram buckets as array-attribute.
def BucketListAttr : TypedArrayAttrBase<HistBucketAttr, "histogram bucket list">;

///
/// Histogram as SPN leaf node.
///
def HistogramNode : HiSPNGraphOp<"histogram",
  [NoSideEffect, DeclareOpInterfaceMethods<LeafNodeInterface>]> {
    let summary = "histogram";

    let description =  [{
        Univariate distribution as leaf node,
        represented by a histogram.
    }];

    let arguments = (ins HiSPNInputType:$index, BucketListAttr:$buckets, UI32Attr:$bucketCount);

    let results = (outs HiSPNProbabilityType);

    let verifier = [{ return ::mlir::spn::high::verify(*this); }];

    let builders = [
        OpBuilderDAG<(ins "Value":$indexVal, "llvm::ArrayRef<std::tuple<int, int, double>>":$buckets)>
    ];
}

///
/// Categorical distribution as SPN leaf node.
///
def CategoricalNode : HiSPNGraphOp<"categorical",
    [NoSideEffect, DeclareOpInterfaceMethods<LeafNodeInterface>]> {

    let summary = "categorical leaf distribution";

    let description = [{
        Univariate categorical leaf distribution.
    }];

    let arguments = (ins HiSPNInputType:$index, F64ArrayAttr:$probabilities);

    let results = (outs HiSPNProbabilityType);

    let verifier = [{ return ::mlir::spn::high::verify(*this); }];

    let builders = [
        OpBuilderDAG<(ins "Value":$indexVal, "llvm::ArrayRef<double>":$probabilities)>
    ];
}

///
/// Gaussian distribution as SPN leaf node.
///
def GaussianNode : HiSPNGraphOp<"gaussian",
    [NoSideEffect, DeclareOpInterfaceMethods<LeafNodeInterface>]> {

    let summary = "Gaussian leaf distribution";

    let description = [{
        Univariate Gaussian leaf distribution.
    }];

    let arguments = (ins HiSPNInputType:$index, F64Attr:$mean, F64Attr:$stddev);

    let results = (outs HiSPNProbabilityType);

    let builders = [
        OpBuilderDAG<(ins "Value":$indexVal, "double":$mean, "double":$stddev)>
    ];

}

#endif // HiSPN_OPS