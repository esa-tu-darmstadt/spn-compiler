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
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/AttrTypeBase.td"

// TODO: What about FunctionLike?
def SPNKernel : LoSPN_Op<"kernel", [FunctionOpInterface, Symbol, IsolatedFromAbove]> {

    let summary = "The Kernel holds the tasks necessary to process an SPN query.";

    let description = [{
        Represents the whole of the computations that need to happen to realize a query.
        The computation itself is represented by one or multiple Tasks associated with the Kernel.
    }];

    let arguments = (ins TypeAttrOf<FunctionType>:$function_type);

    let results = (outs);

    let regions = (region AnyRegion:$body);

    let builders = [
        OpBuilder<(ins "StringRef":$name, "FunctionType":$type)>
    ];
    
    let extraClassDeclaration = [{
        /// Taken from toy Ch2 Opd.td
        /// Does the different prefix naming policy turn getFunctionType() into function_type() ?
        /// This seems to break with some internals of function op interfaces!

        //===------------------------------------------------------------------===//
        // FunctionOpInterface Methods
        //===------------------------------------------------------------------===//

        /// Returns the argument types of this function.
        ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

        /// Returns the result types of this function.
        ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

        /// Returns the number of arguments. Hook for OpTrait::FunctionLike.
        unsigned getNumFuncArguments() { return getFunctionType().getInputs().size(); }

        /// Returns the number of results. Hook for OpTrait::FunctionLike.
        unsigned getNumFuncResults() { return getFunctionType().getResults().size(); }
    }];

    //let verifier = [{ return mlir::spn::low::verifyKernel(*this); }];
    let hasVerifier = 1;
    let skipDefaultBuilders = 0;
}

def SPNTask : LoSPN_Op<"task", [HasParent<"SPNKernel">, SingleBlockImplicitTerminator<"SPNReturn">,
                                RecursiveMemoryEffects, IsolatedFromAbove]> {

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

    //let verifier = [{ return mlir::spn::low::verifyTask(*this); }];
    let hasVerifier = 1;

}

def SPNBody : LoSPN_Op<"body", [HasParent<"SPNTask">, SingleBlockImplicitTerminator<"SPNYield">,
                                NoMemoryEffect, IsolatedFromAbove]> {

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

    //let verifier = [{ return mlir::spn::low::verifyBody(*this); }];
    let hasVerifier = 1;

}

def SPNBatchExtract : LoSPN_Op<"batch_extract", [HasParent<"SPNTask">]> {

    let summary = "Extract a particular index from each sample in a batch";

    let description = [{
        Extract a particular feature for each sample from the input tensor
        representing all features and all samples.

        More specifically,
        ```
        %result = "lo_spn.batch_extract"(%tensor, %dynamicIndex) {staticIndex = $staticIndex, transposed = $transposed}
        ```
        corresponds to
        ```
        if($transposed)
            %result = %tensor[$staticIndex, %dynamicIndex]
        else
            %result = %tensor[%dynamicIndex, $staticIndex]
        ```
        Note how the value of the attribute `$transposed` changes the order of indices.

        Bufferization of the LoSPN dialect will replace this operation with a `batch_read`.
    }];

    let arguments = (ins RankedTensorOf<[LoSPNInputType, LoSPNComputeType]>:$input,
                        Index:$dynamicIndex,
                        UI32Attr:$staticIndex,
                        OptionalAttr<BoolAttr>:$transposed);

    let results = (outs AnyTypeOf<[LoSPNInputType, LoSPNComputeType]>:$result);

    //let verifier = [{ return mlir::spn::low::verifyBatchExtract(*this); }];
    let hasVerifier = 1;
}

def SPNBatchRead : LoSPN_Op<"batch_read", [
                              VectorizableOp,
                              DeclareOpInterfaceMethods<LoSPNVectorizable>]> {

    let summary = "Read a particular index from each sample in a batch represented by a memref";

    let description = [{
        Extract a particular feature for each sample from the input memref
        representing all features and all samples.

        More specifically,
        ```
        %result = "lo_spn.batch_read"(%memref, %dynamicIndex) {staticIndex = $staticIndex, transposed = $transposed}
        ```
        corresponds to
        ```
        if($transposed)
            %result = %memref[$staticIndex, %dynamicIndex]
        else
            %result = %memref[%dynamicIndex, $staticIndex]
        ```
        Note how the value of the attribute `$transposed` changes the order of indices.
    }];

    let arguments = (ins Arg<MemRefOf<[LoSPNInputType, LoSPNComputeType]>, "", [MemRead]>:$batchMem,
                        Index:$dynamicIndex,
                        UI32Attr:$staticIndex,
                        OptionalAttr<BoolAttr>:$transposed);

    let results = (outs AnyTypeOf<[LoSPNInputType, LoSPNComputeType]>:$result);

    //let verifier = [{ return mlir::spn::low::verifyBatchRead(*this); }];
    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "Value":$batchMem, "Value":$dynamicIndex, "unsigned":$staticIndex,
                    "llvm::Optional<bool>":$transposed)>
    ];
}

def SPNBatchCollect : LoSPN_Op<"batch_collect", [HasParent<"SPNTask">]> {

    let summary = "Collect one or multiple results for each sample in a batch";

    let description = [{
        Collect one or multiple results for each sample in a batch into a tensor
        representing all the results from all samples.

        More specifically,
        ```
        %tensor = "lo_spn.batch_collect"(%batchIndex, %result1, ..., %resultN) {transposed = $transposed}
        ```
        corresponds to
        ```
        if($transposed)
            %tensor[1, %batchIndex] = %result1
            ...
            %tensor[N, %batchIndex] = %resultN
        else
            %tensor[%batchIndex, 1] = %result1
            ...
            %tensor[%batchIndex, N] = %resultN
        ```
        Note how the value of the attribute `$transposed` changes the order of indices.

        Bufferization of the LoSPN dialect will replace this operation with a `batch_write`.
    }];

    let arguments = (ins Index:$batchIndex,
                            Variadic<LoSPNComputeType>:$resultValues,
                            OptionalAttr<BoolAttr>:$transposed);

    let results = (outs RankedTensorOf<[LoSPNComputeType]>:$tensor);

    //let verifier = [{ return mlir::spn::low::verifyBatchCollect(*this); }];
    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "ValueRange":$resultValues, "Value":$batchIndex, "bool":$transposed)>
    ];
}

def SPNBatchWrite : LoSPN_Op<"batch_write", [
                                VectorizableOp,
                                DeclareOpInterfaceMethods<LoSPNVectorizable>]> {

    let summary = "Write one or multiple results for each sample in a batch";

    let description = [{
        Store one or multiple results for each sample in a batch into a buffer
        holding all the results from all samples.

        More specifically,
        ```
        "lo_spn.batch_write"(%memref, %batchIndex, %result1, ..., %resultN) {transposed = $transposed}
        ```
        corresponds to
        ```
        if($transposed)
            %memref[1, %batchIndex] = %result1
            ...
            %memref[N, %batchIndex] = %resultN
        else
            %memref[%batchIndex, 1] = %result1
            ...
            %memref[%batchIndex, N] = %resultN
        ```
        Note how the value of the attribute `$transposed` changes the order of indices.
    }];

    let arguments = (ins Arg<MemRefOf<[LoSPNComputeType]>, "", [MemWrite]>:$batchMem,
                        Index:$batchIndex,
                        Variadic<LoSPNComputeType>:$resultValues,
                        OptionalAttr<BoolAttr>:$transposed);

    //let verifier = [{ return mlir::spn::low::verifyBatchWrite(*this); }];
    let hasVerifier = 1;
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

def SPNConvertToScalar : LoSPN_Op<"to_scalar", [NoMemoryEffect]> {
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

class LoSPNBodyOp<string mnemonic, list<Trait> traits = []> : LoSPN_Op<mnemonic,
    !listconcat(traits, [VectorizableOp])>;

def SPNYield : LoSPNBodyOp<"yield", [Terminator, DeclareOpInterfaceMethods<LoSPNVectorizable>]> {

    let summary = "Holds the result computed by the Body";

    let arguments = (ins Variadic<LoSPNComputeType>:$resultValues);

}

def SPNConstant : LoSPNBodyOp<"constant", [ConstantLike, NoMemoryEffect, DeclareOpInterfaceMethods<LoSPNVectorizable>]> {

    let summary = "Represents a simple constant value";

    let arguments = (ins TypeAttr:$type, F64Attr:$value);

    let results = (outs LoSPNComputeType:$result);

    let hasFolder = 1;
}

def SPNLog : LoSPNBodyOp<"log", [NoMemoryEffect, DeclareOpInterfaceMethods<LoSPNVectorizable>]> {

    let summary = "Convert the result of the computation to log";

    let arguments = (ins LoSPNComputeType:$input);

    let results = (outs AnyFloat:$logResult);

    let hasFolder = 1;

}

def SPNMul : LoSPNBodyOp<"mul", [NoMemoryEffect, Commutative, SameOperandsAndResultType,
                                    DeclareOpInterfaceMethods<LoSPNVectorizable>]> {

    let summary = "Multiplication of two values";

    let arguments = (ins LoSPNComputeType:$left, LoSPNComputeType:$right);

    let results = (outs LoSPNComputeType:$result);

    let hasFolder = 1;

}

def SPNAdd : LoSPNBodyOp<"add", [NoMemoryEffect, Commutative, SameOperandsAndResultType,
                                    DeclareOpInterfaceMethods<LoSPNVectorizable>]> {

    let summary = "Addition of two values";

    let arguments = (ins LoSPNComputeType:$left, LoSPNComputeType:$right);

    let results = (outs LoSPNComputeType:$result);

    let hasFolder = 1;

}

/*
/// Named attribute for the inclusive lower bound of a histogram bucket.
def LowerBoundAttr : StructFieldAttr<"lb", I32Attr>;

/// Named attribute for the exclusive upper bound of a histogram bucket.
def UpperBoundAttr : StructFieldAttr<"ub", I32Attr>;

/// Named attribute for the probability value of a histogram bucket.
def HistValueAttr  : StructFieldAttr<"val", F64Attr>;

/// Struct-attribute for a histogram bucket.
def HistBucketAttr : StructAttr<"Bucket", LoSPN_Dialect, [LowerBoundAttr, UpperBoundAttr, HistValueAttr]>;
*/

// TODO
def HistBucketAttr : AttrDef<LoSPN_Dialect, "Bucket"> {
    let summary = "An attribute defining a single histogram bucket";
    let description = "lala";
    let cppClassName = "Bucket";

    let parameters = (ins
        "int": $lb,
        "int": $ub,
        "APFloat": $val
    );

    // inspired by ComplexAttributes.td
    let builders = [
        AttrBuilderWithInferredContext<(ins "int":$lb,
                                            "int":$ub,
                                            "APFloat":$val), [{
            return $_get(nullptr, lb, ub, val);
        }]>
        //AttrBuilderWithInferredContext<(ins)>
    ];

    let genVerifyDecl = 1;
    //let assemblyFormat = "`<` $lb `,` $ub `,` $val `>`";
    let skipDefaultBuilders = 1;
}

/// List of histogram buckets as array-attribute.
def BucketListAttr : TypedArrayAttrBase<HistBucketAttr, "histogram bucket list">;

///
/// Histogram as SPN leaf node.
///
def SPNHistogramLeaf : LoSPNBodyOp<"histogram", [NoMemoryEffect,
                                                 DeclareOpInterfaceMethods<LoSPNVectorizable>,
                                                 DeclareOpInterfaceMethods<LeafNodeInterface>]> {
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
def SPNCategoricalLeaf : LoSPNBodyOp<"categorical", [NoMemoryEffect,
                                                     DeclareOpInterfaceMethods<LoSPNVectorizable>,
                                                     DeclareOpInterfaceMethods<LeafNodeInterface>]> {

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
def SPNGaussianLeaf : LoSPNBodyOp<"gaussian", [NoMemoryEffect,
                                               DeclareOpInterfaceMethods<LoSPNVectorizable>,
                                               DeclareOpInterfaceMethods<LeafNodeInterface>]> {

    let summary = "Gaussian leaf distribution";

    let description = [{
        Univariate Gaussian leaf distribution.
    }];

    let arguments = (ins LoSPNInputType:$index, F64Attr:$mean, F64Attr:$stddev, BoolAttr:$supportMarginal);

    let results = (outs LoSPNComputeType);
}

#endif // LoSPN_Ops