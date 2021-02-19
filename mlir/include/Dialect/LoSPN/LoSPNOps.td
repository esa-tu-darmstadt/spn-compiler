#ifndef LoSPN_OPS
#define LoSPN_OPS

include "LoSPNDialect.td"
include "LoSPNInterfaces.td"
include "LoSPNTraits.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"

def SPNKernel : LoSPN_Op<"kernel", [FunctionLike, Symbol, IsolatedFromAbove]> {

    let summary = "The Kernel holds the tasks necessary to process an SPN query.";

    let description = [{
        Represents the whole of the computations that need to happen to realize a query.
        The computation itself is represented by one or multiple Tasks associated with the Kernel.
    }];

    // TODO: Rename kernelType to type?
    let arguments = (ins TypeAttr:$kernelType, StrAttr:$kernelName);

    let results = (outs);

    let regions = (region AnyRegion:$body);

    let builders = [
        OpBuilderDAG<(ins "StringRef":$name, "FunctionType":$type)>
    ];

    let extraClassDeclaration = [{
      private:
        // This trait needs access to the hooks defined below.
        friend class OpTrait::FunctionLike<SPNKernel>;

        /// Returns the number of arguments. Hook for OpTrait::FunctionLike.
        unsigned getNumFuncArguments() { return getType().getNumInputs(); }

        /// Returns the number of results. Hook for OpTrait::FunctionLike.
        unsigned getNumFuncResults() { return getType().getNumResults(); }

    }];
}

def SPNTask : LoSPN_Op<"task", [HasParent<"SPNKernel">]> {

    let summary = "The task holds (part of) the computation necessary for SPN query processing";

    let description = [{
        Represents a part of the computation inside a Kernel and is therefore directly associated with the Kernel.
        The Task has a Body associated with it, which represents the computation for a single sample/input, while
        the Task represents what is necessary to compute the result for all samples in the input tensor.
    }];

    let arguments = (ins Variadic<AnyTypeOf<[RankedTensorOf<[LoSPNInputType, LoSPNComputeType]>,
                                                MemRefOf<[LoSPNInputType, LoSPNComputeType]>]>>:$inputs,
                        UI32Attr:$batchSize);

    let results = (outs Variadic<RankedTensorOf<[LoSPNComputeType]>>:$results);

    let regions = (region SizedRegion<1>:$body);

}

def SPNBody : LoSPN_Op<"body", [HasParent<"SPNTask">]> {

    let summary = "The body holds the the actual computational operations";

    let description = [{
        Represents the computation for a single input and contains the actual computation operations as child nodes.
    }];

    let arguments = (ins Variadic<AnyTypeOf<[LoSPNInputType, LoSPNComputeType]>>:$inputs);

    let results = (outs Variadic<LoSPNComputeType>:$outputs);

    let regions = (region SizedRegion<1>:$body);

}

def SPNBatchExtract : LoSPN_Op<"batch_extract", [HasParent<"SPNTask">]> {

    let summary = "Extract a particular index from each sample in a batch";

    let arguments = (ins RankedTensorOf<[LoSPNInputType, LoSPNComputeType]>:$input,
                        Index:$batchIndex,
                        UI32Attr:$sampleIndex);

    let results = (outs AnyTypeOf<[LoSPNInputType, LoSPNComputeType]>:$result);

}

def SPNBatchRead : LoSPN_Op<"batch_read", [
                              HasParent<"SPNTask">,
                              VectorizableOp,
                              DeclareOpInterfaceMethods<LoSPNVectorizable>]> {

    let summary = "Read a particular index from each sample in a batch represented by  a memref";

    let arguments = (ins MemRefOf<[LoSPNInputType, LoSPNComputeType]>:$batchMem,
                        Index:$batchIndex,
                        UI32Attr:$sampleIndex);

    let results = (outs AnyTypeOf<[LoSPNInputType, LoSPNComputeType]>:$result);

}

def SPNBatchCollect : LoSPN_Op<"batch_collect", [HasParent<"SPNTask">, Terminator]> {

    let summary = "Collect one or multiple results for each sample in a batch";

    let arguments = (ins Variadic<LoSPNComputeType>:$results,
                            Index:$batchIndex);

    let results = (outs Variadic<RankedTensorOf<[LoSPNComputeType]>>:$tensors);

}

def SPNBatchWrite : LoSPN_Op<"batch_write", [
                                HasParent<"SPNTask">,
                                VectorizableOp,
                                DeclareOpInterfaceMethods<LoSPNVectorizable>]> {

    let summary = "Write one or multiple results for each sample in a batch";

    let arguments = (ins Variadic<LoSPNComputeType>:$results,
                        MemRefOf<[LoSPNComputeType]>:$batchMem,
                        Index:$batchIndex);
}

def SPNCopy : LoSPN_Op<"copy", []> {

    let summary = "Copy content of one memref to another memref";

    /// TODO Detailed description

    let arguments = (ins MemRefOf<[LoSPNComputeType]>:$source,
                        MemRefOf<[LoSPNComputeType]>:$target);

}

def SPNConvertToVector : LoSPN_Op<"to_vector", []> {
    let summary = "Convert scalar input to vector";

    let description = [{
        Converts from a single scalar value a vector.
        This operation is only used as a materialization during type conversion
        and does not generate actual code to convert the value to a vector.
    }];

    let arguments = (ins LoSPNComputeType:$input);

    let results = (outs VectorOf<[LoSPNComputeType]>:$vector);

}

def SPNReturn : LoSPN_Op<"return", [Terminator]> {

    let summary = "Return one or multiple results from a Task or Kernel";

    let arguments = (ins Variadic<RankedTensorOf<[LoSPNComputeType]>>:$returnValues);
}

class LoSPNBodyOp<string mnemonic, list<OpTrait> traits = []> : LoSPN_Op<mnemonic,
    !listconcat(traits, [HasParent<"SPNBody">, VectorizableOp, DeclareOpInterfaceMethods<LoSPNVectorizable>])>;

def SPNYield : LoSPNBodyOp<"yield", [Terminator]> {

    let summary = "Holds the result computed by the Body";

    let arguments = (ins Variadic<LoSPNComputeType>:$resultValues);

}

def SPNConstant : LoSPNBodyOp<"constant", [ConstantLike, NoSideEffect]> {

    let summary = "Represents a simple constant value";

    let arguments = (ins TypeAttr:$type, F64Attr:$value);

    let results = (outs LoSPNComputeType:$result);

    let hasFolder = 1;
}

def SPNLog : LoSPNBodyOp<"log", [NoSideEffect]> {

    let summary = "Convert the result of the computation to log";

    let arguments = (ins LoSPNComputeType:$input);

    let results = (outs F64:$logResult);

}

def SPNMul : LoSPNBodyOp<"mul", [NoSideEffect, Commutative, SameOperandsAndResultType]> {

    let summary = "Multiplication of two values";

    let arguments = (ins LoSPNComputeType:$left, LoSPNComputeType:$right);

    let results = (outs LoSPNComputeType:$result);

}

def SPNAdd : LoSPNBodyOp<"add", [NoSideEffect, Commutative, SameOperandsAndResultType]> {

    let summary = "Addition of two values";

    let arguments = (ins LoSPNComputeType:$left, LoSPNComputeType:$right);

    let results = (outs LoSPNComputeType:$result);

}

/// Named attribute for the inclusive lower bound of a histogram bucket.
def LowerBoundAttr : StructFieldAttr<"lb", I32Attr>;

/// Named attribute for the exclusive upper bound of a histogram bucket.
def UpperBoundAttr : StructFieldAttr<"ub", I32Attr>;

/// Named attribute for the probability value of a histogram bucket.
def HistValueAttr  : StructFieldAttr<"val", F64Attr>;

/// Struct-attribute for a histogram bucket.
def HistBucketAttr : StructAttr<"Bucket", LoSPN_Dialect, [LowerBoundAttr, UpperBoundAttr, HistValueAttr]>;

/// List of histogram buckets as array-attribute.
def BucketListAttr : TypedArrayAttrBase<HistBucketAttr, "histogram bucket list">;

///
/// Histogram as SPN leaf node.
///
def SPNHistogramLeaf : LoSPNBodyOp<"histogram", [NoSideEffect]> {
    let summary = "histogram";

    let description =  [{
        Univariate distribution as leaf node,
        represented by a histogram.
    }];

    let arguments = (ins LoSPNInputType:$index, BucketListAttr:$buckets,
                        UI32Attr:$bucketCount, BoolAttr:$supportMarginal);

    let results = (outs LoSPNComputeType);

}

///
/// Categorical distribution as SPN leaf node.
///
def SPNCategoricalLeaf : LoSPNBodyOp<"categorical", [NoSideEffect]> {

    let summary = "categorical leaf distribution";

    let description = [{
        Univariate categorical leaf distribution.
    }];

    let arguments = (ins LoSPNInputType:$index, F64ArrayAttr:$probabilities, BoolAttr:$supportMarginal);

    let results = (outs LoSPNComputeType);
}

///
/// Gaussian distribution as SPN leaf node.
///
def SPNGaussianLeaf : LoSPNBodyOp<"gaussian", [NoSideEffect]> {

    let summary = "Gaussian leaf distribution";

    let description = [{
        Univariate Gaussian leaf distribution.
    }];

    let arguments = (ins LoSPNInputType:$index, F64Attr:$mean, F64Attr:$stddev, BoolAttr:$supportMarginal);

    let results = (outs LoSPNComputeType);
}

#endif // LoSPN_Ops