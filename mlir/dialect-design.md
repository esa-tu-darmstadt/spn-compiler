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
  computing the result. The computation itself is represented by one or multiple **Task**s associated with the **
  Kernel**.

* **Task**: Represents a part of the computation inside a **Kernel** and is therefore directly associated with the **
  Kernel**. The task also takes tensors as input and produces tensors as outputs, which are later also bufferized.
  The **Task** has a **Body** associated with it, which represents the computation for a single sample/input, while
  the **Task** represents what is necessary to compute the result for all samples in the input tensor and therefore gets
  lowered to constructs such as loops or GPU kernels.

* **Body**: Represents the computation for a single input and contains the actual computation operations as child nodes.

* **Add**: Two-input addition of two values, or element-wise addition of two vectors.

* **Mul**: Two-input multiplication of two values, or element-wise multiplication of two vectors.

* **Constant**: Constant value.

* **HistogramOp**: Computational realization of the **HistogramLeaf**.

* **CategoricalOp**: Computational realization of the **CategoricalLeaf**.

* **GaussianOp**: Computational realization of the **GaussianLeaf**.

* **Result**: Holds the one or multiple results for this body, terminator.