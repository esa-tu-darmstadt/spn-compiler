# SPN Dialect Design #

The compiler uses two main dialects to represent SPNs:

* **HiSPN** abstractly represents a query on a given Sum-Product Network.

* **LoSPN** specifies how to actually conduct the computation to materialize the requested query on the given SPN.

The dialect conversion from **HiSPN** to **LoSPN** represents the knowledge about which computation needs to be
performed to actually evaluate the requested query.

**LoSPN** represents the computation and is then successively lowered to more concrete realizations of the computation,
specifying which computations, loops, etc. need to be performed on which data. As part of this process, optimizations
such as vectorization, graph-partitioning etc. can then happen.

## HiSPN ##

### Operations ###

* **Query**: There is no concrete QueryOp, but rather this is an abstract concept implemented by the various concrete
  queries. For access to some common properties of all queries, the **QueryInterface** is available. A query typically
  stores some information about the query as attributes and has one or multiple
  **SPNGraph** associated with it as its only operations.

* **JointProbability**: Represents a joint probability query. Has a single **SPNGraph** associated with it.

* **SPNGraph**: Holds the actual Sum-Product Network inside its single region and is directly associated with a **
  Query**. The single block in the region has a block argument for each feature of the SPN.

* **Leaf**: Again, this is more of an abstract concept than a concrete operation, realized via the **LeafNodeInterface**
  .

* **HistogramLeaf**

* **CategoricalLeaf**

* **GaussianLeaf**

* **Sum** N-ary weighted sum with a number of operands and weights (as attributes) associated with it.

* **Product** N-ary product of a number of operands.

* **Root** The root of the Sum-Product Network, terminator.

## LoSPN ##

### Operations ###

* **Kernel**: Represents the whole of the computations that need to happen to realize a query. Takes a tensor as input
  and produces a tensor as output. These tensors are later bufferized to eventually yield a function taking inputs and
  computing the result. The computation itself is represented by one or multiple **Task**s associated with the
  **Kernel**.

* **Task**: Represents a part of the computation inside a **Kernel** and is therefore directly associated with the **
  Kernel**. The task also takes tensors as input and produces tensors as outputs, which are later also bufferized.
  The **Task** has a **Body** associated with it, which represents the computation for a single sample/input, while
  the **Task** represents what is necessary to compute the result for all samples in the input tensor and therefore gets
  lowered to constructs such as loops or GPU kernels. The first argument of the first block of task always represents
  the index of the currently processed sample in a batch.

* **BatchExtract/BatchRead**: Extracts a single feature value from the currently processed sample in the batch.
  **BatchExtract** is used for tensors, **BatchRead** is used for memrefs after bufferization.

* **BatchCollect/BatchWrite**: Collect results for each sample in a batch into a tensor/memref. **BatchCollect**
  is used for tensors, **BatchWrite** for memrefs after bufferization.

* **Body**: Represents the computation for a single input and contains the actual computation operations as child nodes.

* **Add**: Two-input addition of two values, or element-wise addition of two vectors.

* **Mul**: Two-input multiplication of two values, or element-wise multiplication of two vectors.

* **Constant**: Constant value.

* **HistogramOp**: Computational realization of the **HistogramLeaf**.

* **CategoricalOp**: Computational realization of the **CategoricalLeaf**.

* **GaussianOp**: Computational realization of the **GaussianLeaf**.

* **Yield**: Holds the one or multiple results for this body, terminator.

## Lowering & Transformations ##

### HiSPN to LoSPN ###

The lowering from HiSPN to LoSPN is a two-stage process:

1. In a first step, the nodes of the Graph are lowered. The processes uses the semantic of the surrounding query to
   determine which LoSPN operations are required to implement a node in the Graph.

2. In the second step, the query itself is transformed into a Kernel containing the corresponding Tasks.

### LoSPN Transformations ###

#### Bufferization ####

During bufferization, tensor semantic is transformed into memory access semantic. This step should only happen after all
transformations that can benefit from the simpler return-tensors semantic (e.g., graph partitioning)
have been performed. During bufferization, all Tensor inputs are transformed into MemRef inputs and all Tensor results
are transformed into MemRef **out-args**. The latter also involves allocating temporary memories. This transformation is
performed on the Kernel and all its Tasks.

#### Canonicalization ####

Uses the MLIR canonicalization framework. Arithmetic operations inside the Body (e.g., Log, Add, Mul) are constant
folded where possible.

#### Common Subexpression Elimination (CSE) #####

Uses the regular CSE pass from the MLIR framework to eliminate redundant arithmetic subexpressions in the Body.

### LoSPN to CPU ###

Lowers from LoSPN to a mixture of Standard, Math, SCF (structured control flow) and, optionally, Vector dialect,
targeting the native CPU target. This also is a multi-stage process:

1. Lower the structure, i.e., Kernels and Tasks into functions and control-flow. Kernels and Tasks are both lowered to
   functions, and the calls to Tasks are inserted into the Kernel. Inside the Task function, loops are used to process
   the whole batch. If vectorization is enabled, Tasks for which all operations inside the body can be vectorized, will
   be transformed into vectorized loops (and an epilog loop) and the operations inside the vectorized loop are marked as
   such using the `Vectorizable` interface/trait.

2. If vectorization was enabled, all operations (nodes) that were marked vectorized are now lowered.

3. The remaining (scalar) operations are also lowered. While the previous two conversion were only partial conversions,
   this is a full conversion and requires all LoSPN operations to be legalized. This is an analogy to the process used
   for bufferization in MLIR, where a final(izing) conversion completes the process.

### CPU Transformations ###

#### Finalize Bufferization ####

As the bufferization passes for the up-stream dialects use functions as entry points, they can only be executed after
the conversion to Standard dialect. Here, the tensor-bufferization and the finalization of the bufferization take place,
removing all previously introduced materializations.

#### Buffer Deallocation ####

Inserts deallocations for the temporary buffers that were introduced during the LoSPN bufferization of Tasks.

#### Copy Removal ####

Removes unnecessary copy operations from one buffer to the other, e.g., from a local buffer to the out-arg of the
Kernel.