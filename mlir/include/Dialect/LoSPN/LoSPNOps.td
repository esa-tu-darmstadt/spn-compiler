//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

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

    let results = (outs);

    let regions = (region AnyRegion:$body);

    let builders = [
        OpBuilder<(ins "StringRef":$name, "FunctionType":$type)>
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

    let verifier = [{ return mlir::spn::low::verifyKernel(*this); }];
}

def SPNTask : LoSPN_Op<"task", [HasParent<"SPNKernel">, SingleBlockImplicitTerminator<"SPNReturn">,
                                RecursiveSideEffects, IsolatedFromAbove]> {

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

    let extraClassDeclaration = [{
        public:
            /// Add an entry block to the task. The corresponding block arguments
            /// are automatically added to the block.
            Block* addEntryBlock();

            /// Get the block argument corresponding to the batch index.
            Value getBatchIndex();
    }];

    let verifier = [{ return mlir::spn::low::verifyTask(*this); }];

}

def SPNBody : LoSPN_Op<"body", [HasParent<"SPNTask">, SingleBlockImplicitTerminator<"SPNYield">,
                                NoSideEffect, IsolatedFromAbove]> {

    let summary = "The body holds the the actual computational operations";

    let description = [{
        Represents the computation for a single input and contains the actual computation operations as child nodes.
    }];

    let arguments = (ins Variadic<AnyTypeOf<[LoSPNInputType, LoSPNComputeType]>>:$inputs);

    let results = (outs Variadic<LoSPNComputeType>:$outputs);

    let regions = (region SizedRegion<1>:$body);

    let extraClassDeclaration = [{
        public:
            /// Add an entry block to the body, the corresponding block
            /// arguments are automatically added to the block.
            Block* addEntryBlock();
    }];

    let verifier = [{ return mlir::spn::low::verifyBody(*this); }];

}

def SPNBatchExtract : LoSPN_Op<"batch_extract", [HasParent<"SPNTask">]> {

    let summary = "Extract a particular index from each sample in a batch";

    let arguments = (ins RankedTensorOf<[LoSPNInputType, LoSPNComputeType]>:$input,
                        Index:$batchIndex,
                        UI32Attr:$sampleIndex);

    let results = (outs AnyTypeOf<[LoSPNInputType, LoSPNComputeType]>:$result);

    let verifier = [{ return mlir::spn::low::verifyBatchExtract(*this); }];
}

def SPNBatchRead : LoSPN_Op<"batch_read", [
                              VectorizableOp,
                              DeclareOpInterfaceMethods<LoSPNVectorizable>]> {

    let summary = "Read a particular index from each sample in a batch represented by  a memref";

    let arguments = (ins Arg<MemRefOf<[LoSPNInputType, LoSPNComputeType]>, "", [MemRead]>:$batchMem,
                        Index:$batchIndex,
                        UI32Attr:$sampleIndex);

    let results = (outs AnyTypeOf<[LoSPNInputType, LoSPNComputeType]>:$result);

    let verifier = [{ return mlir::spn::low::verifyBatchRead(*this); }];
}

def SPNBatchCollect : LoSPN_Op<"batch_collect", [HasParent<"SPNTask">]> {

    let summary = "Collect one or multiple results for each sample in a batch";

    let arguments = (ins Variadic<LoSPNComputeType>:$resultValues,
                            Index:$batchIndex);

    let results = (outs Variadic<RankedTensorOf<[LoSPNComputeType]>>:$tensors);

    let verifier = [{ return mlir::spn::low::verifyBatchCollect(*this); }];
}

def SPNBatchWrite : LoSPN_Op<"batch_write", [
                                VectorizableOp,
                                DeclareOpInterfaceMethods<LoSPNVectorizable>]> {

    let summary = "Write one or multiple results for each sample in a batch";

    let arguments = (ins LoSPNComputeType:$resultValue,
                        Arg<MemRefOf<[LoSPNComputeType]>, "", [MemWrite]>:$batchMem,
                        Index:$batchIndex);

    let verifier = [{ return mlir::spn::low::verifyBatchWrite(*this); }];
}

def SPNCopy : LoSPN_Op<"copy", [SameOperandsShape]> {

    let summary = "Copy content of one memref to another memref";

    let description = [{
        Copies the content of one memref to another memref.
        This operation is used as part of the bufferization
        process for the LoSPN dialect, where Tasks write
        their results to previously allocated MemRefs instead
        of returning Tensors.
    }];

    let arguments = (ins Arg<MemRefOf<[LoSPNComputeType]>, "", [MemRead]>:$source,
                        Arg<MemRefOf<[LoSPNComputeType]>, "", [MemWrite]>:$target);

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

def SPNConvertToScalar : LoSPN_Op<"to_scalar", [NoSideEffect]> {
    let summary = "Convert vector input to scalar";

    let description = [{
        Converts from a vector to a single scalar.
        This operation is only used as a materialization during type conversion
        and does not generate actual code to convert the value to a scalar.
    }];

    let arguments = (ins VectorOf<[LoSPNComputeType]>:$vector);

    let results = (outs LoSPNComputeType:$output);

}

def SPNStripLog : LoSPN_Op<"strip_log", [VectorizableOp, DeclareOpInterfaceMethods<LoSPNVectorizable>]> {
    let summary = "Strip the log-space property from a value";

    let description = [{
        Remove the log-space property from a value.
        This operation is only used as a materialization during type
        conversion and does not generate actual code.
    }];

    let arguments = (ins LoSPN_LogType:$input, TypeAttr:$target);

    let results = (outs LoSPNComputeType:$out);

    let builders = [
        OpBuilder<(ins "Value":$input, "Type":$targetType)>
    ];
}

def SPNConvertLog : LoSPN_Op<"convert_log", [VectorizableOp, DeclareOpInterfaceMethods<LoSPNVectorizable>]> {
    let summary = "Add the log-space property to a value";

    let description = [{
        Add the log-space property to a value.
        This operation is only used as a materialization during type
        conversion and does not generate actual code.
    }];

    let arguments = (ins LoSPNComputeType:$input);

    let results = (outs LoSPN_LogType:$out);

    let builders = [
        OpBuilder<(ins "Value":$input)>
    ];

}

def SPNReturn : LoSPN_Op<"return", [Terminator]> {

    let summary = "Return one or multiple results from a Task or Kernel";

    let arguments = (ins Variadic<RankedTensorOf<[LoSPNComputeType]>>:$returnValues);
}

class LoSPNBodyOp<string mnemonic, list<OpTrait> traits = []> : LoSPN_Op<mnemonic,
    !listconcat(traits, [VectorizableOp])>;

def SPNYield : LoSPNBodyOp<"yield", [Terminator, DeclareOpInterfaceMethods<LoSPNVectorizable>]> {

    let summary = "Holds the result computed by the Body";

    let arguments = (ins Variadic<LoSPNComputeType>:$resultValues);

}

def SPNConstant : LoSPNBodyOp<"constant", [ConstantLike, NoSideEffect, DeclareOpInterfaceMethods<LoSPNVectorizable>]> {

    let summary = "Represents a simple constant value";

    let arguments = (ins TypeAttr:$type, F64Attr:$value);

    let results = (outs LoSPNComputeType:$result);

    let hasFolder = 1;
}

def SPNLog : LoSPNBodyOp<"log", [NoSideEffect, DeclareOpInterfaceMethods<LoSPNVectorizable>]> {

    let summary = "Convert the result of the computation to log";

    let arguments = (ins LoSPNComputeType:$input);

    let results = (outs AnyFloat:$logResult);

    let hasFolder = 1;

}

def SPNMul : LoSPNBodyOp<"mul", [NoSideEffect, Commutative, SameOperandsAndResultType,
                                    DeclareOpInterfaceMethods<LoSPNVectorizable>]> {

    let summary = "Multiplication of two values";

    let arguments = (ins LoSPNComputeType:$left, LoSPNComputeType:$right);

    let results = (outs LoSPNComputeType:$result);

    let hasFolder = 1;

}

def SPNAdd : LoSPNBodyOp<"add", [NoSideEffect, Commutative, SameOperandsAndResultType,
                                    DeclareOpInterfaceMethods<LoSPNVectorizable>]> {

    let summary = "Addition of two values";

    let arguments = (ins LoSPNComputeType:$left, LoSPNComputeType:$right);

    let results = (outs LoSPNComputeType:$result);

    let hasFolder = 1;

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
def SPNHistogramLeaf : LoSPNBodyOp<"histogram", [NoSideEffect,
                                                 DeclareOpInterfaceMethods<LoSPNVectorizable>,
                                                 DeclareOpInterfaceMethods<LeafNodeInterface>]> {
    let summary = "histogram";

    let description =  [{
        Univariate distribution as leaf node,
        represented by a histogram.
    }];

    let arguments = (ins LoSPNInputType:$index, BucketListAttr:$buckets,
                        UI32Attr:$bucketCount, BoolAttr:$supportMarginal);

    let hasCanonicalizeMethod = 1;

    let results = (outs LoSPNComputeType);
}

///
/// Categorical distribution as SPN leaf node.
///
def SPNCategoricalLeaf : LoSPNBodyOp<"categorical", [NoSideEffect,
                                                     DeclareOpInterfaceMethods<LoSPNVectorizable>,
                                                     DeclareOpInterfaceMethods<LeafNodeInterface>]> {

    let summary = "categorical leaf distribution";

    let description = [{
        Univariate categorical leaf distribution.
    }];

    let arguments = (ins LoSPNInputType:$index, F64ArrayAttr:$probabilities, BoolAttr:$supportMarginal);

    let hasCanonicalizeMethod = 1;

    let results = (outs LoSPNComputeType);
}

///
/// Gaussian distribution as SPN leaf node.
///
def SPNGaussianLeaf : LoSPNBodyOp<"gaussian", [NoSideEffect,
                                               DeclareOpInterfaceMethods<LoSPNVectorizable>,
                                               DeclareOpInterfaceMethods<LeafNodeInterface>]> {

    let summary = "Gaussian leaf distribution";

    let description = [{
        Univariate Gaussian leaf distribution.
    }];

    let arguments = (ins LoSPNInputType:$index, F64Attr:$mean, F64Attr:$stddev, BoolAttr:$supportMarginal);

    let results = (outs LoSPNComputeType);
}

///
/// Select of an SPN leaf node value.
/// Corresponds to: ($input < $input_true_threshold) ? $val_true : $val_false;
///
def SPNSelectLeaf : LoSPNBodyOp<"select", [NoSideEffect,
                                           DeclareOpInterfaceMethods<LoSPNVectorizable>,
                                           DeclareOpInterfaceMethods<LeafNodeInterface>]> {

    let summary = "Leaf node value select";

    let description = [{
        Single value select of a Categorical or Histogram leaf.
    }];

    let arguments = (ins LoSPNInputType:$input, F64Attr:$input_true_threshold,
        F64Attr:$val_true, F64Attr:$val_false, BoolAttr:$supportMarginal);

    let results = (outs LoSPNComputeType);

}

#endif // LoSPN_Ops