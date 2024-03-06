//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef HiSPN_OPS
#define HiSPN_OPS

include "HiSPNDialect.td"
include "HiSPNInterfaces.td"
include "HiSPNAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/RegionKindInterface.td"

///
/// Attributes representing different kinds of arithmetic errors:
///
def AbsoluteErrorAttr : I32EnumAttrCase<"absolute_error", 1>;
def RelativeErrorAttr : I32EnumAttrCase<"relative_error", 2>;
def AbsoluteLogErrorAttr : I32EnumAttrCase<"absolute_log_error", 3>;
def ErrorModelAttr : I32EnumAttr<"error_model", "Arithmetic error kind",
    [AbsoluteErrorAttr, RelativeErrorAttr, AbsoluteLogErrorAttr]> {
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
                    TypeAttr:$featureDataType,
                    SymbolNameAttr:$queryName,
                    UI32Attr:$batchSize,
                    ErrorModelAttr:$errorModel,
                    F64Attr:$maxError,
                    BoolAttr:$supportMarginal
                    );

    let regions = (region SizedRegion<1>:$graph);

    let hasVerifier = true;
}

///
/// Operation representing the actual directed acyclic graph making up the Sum-Product Network.
///
def Graph : HiSPN_Op<"graph",
    [DeclareOpInterfaceMethods<RegionKindInterface>, IsolatedFromAbove,
    // TODO Use ParentOneOf<list<string> opList> as soon as we have multiple queries
    HasParent<"JointQuery">,
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
class HiSPNGraphOp<string mnemonic, list<Trait> traits = []> : HiSPN_Op<mnemonic,
    !listconcat(traits, [HasParent<"Graph">, DeclareOpInterfaceMethods<GraphNodeInterface>])>;

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
def ProductNode : HiSPNGraphOp<"product", [NoMemoryEffect, SameOperandsAndResultType]> {
    let summary = "N-Ary SPN Product node";

    let arguments = (ins Variadic<HiSPNProbabilityType>:$operands);

    let results = (outs HiSPNProbabilityType);

    let hasVerifier = true;
}

///
/// N-ary Sum node inside the SPN DAG.
///
def SumNode : HiSPNGraphOp<"sum", [NoMemoryEffect, SameOperandsAndResultType]> {
    let summary = "Weighted N-ary SPN Sum node";

    let description = [{
        N-ary, weighted sum operation, multiplying each
        operand with the associated, constant weight
        and adding all operands into a single result.
    }];

    let arguments = (ins Variadic<HiSPNProbabilityType>:$operands, F64ArrayAttr:$weights);

    let results = (outs HiSPNProbabilityType);

    let hasVerifier = true;

    let builders = [
        OpBuilder<(ins "llvm::ArrayRef<Value>":$operands, "llvm::ArrayRef<double>":$weights)>
    ];

}

///
/// Histogram as SPN leaf node.
///
def HistogramNode : HiSPNGraphOp<"histogram",
  [NoMemoryEffect, DeclareOpInterfaceMethods<LeafNodeInterface>]> {
    let summary = "histogram";

    let description =  [{
        Univariate distribution as leaf node,
        represented by a histogram.
    }];

    let arguments = (ins HiSPNInputType:$index, HistBucketList:$buckets, UI32Attr:$bucketCount);

    let results = (outs HiSPNProbabilityType);

    let hasVerifier = true;

    let builders = [
        OpBuilder<(ins "Value":$indexVal, "llvm::ArrayRef<std::tuple<int, int, APFloat>>":$buckets)>
    ];
}

///
/// Categorical distribution as SPN leaf node.
///
def CategoricalNode : HiSPNGraphOp<"categorical",
    [NoMemoryEffect, DeclareOpInterfaceMethods<LeafNodeInterface>]> {

    let summary = "categorical leaf distribution";

    let description = [{
        Univariate categorical leaf distribution.
    }];

    let arguments = (ins HiSPNInputType:$index, F64ArrayAttr:$probabilities);

    let results = (outs HiSPNProbabilityType);

    let hasVerifier = true;

    let builders = [
        OpBuilder<(ins "Value":$indexVal, "llvm::ArrayRef<double>":$probabilities)>
    ];
}

///
/// Gaussian distribution as SPN leaf node.
///
def GaussianNode : HiSPNGraphOp<"gaussian",
    [NoMemoryEffect, DeclareOpInterfaceMethods<LeafNodeInterface>]> {

    let summary = "Gaussian leaf distribution";

    let description = [{
        Univariate Gaussian leaf distribution.
    }];

    let arguments = (ins HiSPNInputType:$index, F64Attr:$mean, F64Attr:$stddev);

    let results = (outs HiSPNProbabilityType);

    let builders = [
        OpBuilder<(ins "Value":$indexVal, "double":$mean, "double":$stddev)>
    ];

}

#endif // HiSPN_OPS