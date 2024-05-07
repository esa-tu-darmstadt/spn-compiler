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
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"


def SPNKernel : LoSPN_Op<"kernel", [CallableOpInterface, FunctionOpInterface, Symbol, IsolatedFromAbove]> {

    let summary = "The Kernel holds the tasks necessary to process an SPN query.";

    let description = [{
        Represents the whole of the computations that need to happen to realize a query.
        The computation itself is represented by one or multiple Tasks associated with the Kernel.
    }];

    let results = (outs);

    let regions = (region AnyRegion:$body);
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );

    
    let builders = [OpBuilder<(ins
        "StringRef":$name, "FunctionType":$type,
        CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
    >];

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // CallableOpInterface
        //===------------------------------------------------------------------===//

        /// Returns the region on the current operation that is callable. This may
        /// return null in the case of an external callable object, e.g. an external
        /// function.
        ::mlir::Region *getCallableRegion() {
        return isExternal() ? nullptr : &getBody();
        }

        /// Returns the results types that the callable region produces when
        /// executed.
        ArrayRef<Type> getCallableResults() {
        return getFunctionType().getResults();
        }

        /// Returns the argument attributes for all callable region arguments or
        /// null if there are none.
        ::mlir::ArrayAttr getCallableArgAttrs() {
        return getArgAttrs().value_or(nullptr);
        }

        /// Returns the result attributes for all callable region results or
        /// null if there are none.
        ::mlir::ArrayAttr getCallableResAttrs() {
        return getResAttrs().value_or(nullptr);
        }

        //===------------------------------------------------------------------===//
        // FunctionOpInterface Methods
        //===------------------------------------------------------------------===//

        /// Returns the argument types of this function.
        ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

        /// Returns the result types of this function.
        ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    }];

    let hasVerifier = true;
    let skipDefaultBuilders = 1;
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

    let hasVerifier = true;

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

    let hasVerifier = true;

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

    let hasVerifier = true;
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

    let hasVerifier = true;

    let builders = [
        OpBuilder<(ins "Value":$batchMem, "Value":$dynamicIndex, "unsigned":$staticIndex,
                    "std::optional<bool>":$transposed)>
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

    let hasVerifier = true;

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

    let hasVerifier = true;
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

def SPNConstant : LoSPNBodyOp<"constant", [ConstantLike, Pure, 
    TypesMatchWith<"result type matches value type, ignoring a log type wrapper if present",
                   "result", "value", 
                   "($_self.isa<LogType>() ? $_self.cast<LogType>().getBaseType() : $_self)">, 
    DeclareOpInterfaceMethods<LoSPNVectorizable>]> {

    let summary = "Represents a simple constant value";

    let arguments = (ins TypedAttrInterface:$value);

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

    let arguments = (ins LoSPNInputType:$index, ArrayAttr:$buckets,
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