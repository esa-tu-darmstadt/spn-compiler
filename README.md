# SPN Compiler #

## About SPNC ##

**SPNC** is a multi-target compiler for Sum-Product Networks, a class of machine learning models.

Starting with release 0.0.4, **SPNC** is mostly completely implemented in `C++` and uses the [LLVM compiler framework](https://llvm.org/) 
and [MLIR](https://mlir.llvm.org) for code generation for the different targets.

Currently supported targets are CPUs (all architecture supported by LLVM, vectorization currently limited to X86) and CUDA GPUs.


### Installation ###

**SPNC** comprises two main parts: `xspn`, a small library to help with the serialization of SPFlow models, and `spnc`, which is the compiler itself.

The easiest way to install both components is to use the pre-built Python packages (wheels) provided on the SPNC Github page. 
While `xspn` is completely platform-independent, the pre-built wheels for `spnc` only works on Linux platforms. 
See the [installation instructions](https://github.com/esa-tu-darmstadt/spn-compiler/wiki/Installation) for detailed 
requirements. 

In case you want to use **SPNC** on a different platform or want to build **SPNC** from source,
follow the [installation instructions](https://github.com/esa-tu-darmstadt/spn-compiler/wiki/Installation) to build 
**SPNC** and all its dependencies from source. Currently, `spnc` is based on LLVM commit `f8d3f47e1fd09392aa30df83849b25acd8c59a25`.

### Usage ###

**SPNC** was designed to directly interact with [SPFlow](https://spflow.github.io/SPFlow/), 
a library for SPN modeling and training. 

The Python interface of `spnc` allows to directly process SPNs created in SPFlow 
(see the SPFlow manuals for more information on construction of SPNs).

The inference from SPFlow can directly be replaced with invocations of the compiler, which will 
compile the SPN for fast inference and perform inference by executing the compiled kernel. 

The following example shows how to invoke the inference through the compiler for a small example SPN:

```python
import numpy as np

from spn.structure.Base import Product, Sum
from spn.structure.leaves.parametric.Parametric import Categorical
from spn.algorithms.Inference import log_likelihood

from spnc.cpu import CPUCompiler

# Construct a small SPN
c1 = Categorical(p=[0.35, 0.55, 0.1], scope=0)
c2 = Categorical(p=[0.25, 0.625, 0.125], scope=1)
c3 = Categorical(p=[0.5, 0.2, 0.3], scope=2)
c4 = Categorical(p=[0.6, 0.15, 0.25], scope=3)
c5 = Categorical(p=[0.7, 0.11, 0.19], scope=4)
c6 = Categorical(p=[0.8, 0.14, 0.06], scope=5)
p = Product(children=[c1, c2, c3, c4, c5, c6])

# Create some random input values.
inputs = np.column_stack((
    np.random.randint(3, size=30),
    np.random.randint(3, size=30),
    np.random.randint(3, size=30),
    np.random.randint(3, size=30),
    np.random.randint(3, size=30),
    np.random.randint(3, size=30),
)).astype("int32")

# Compile the SPN and execute inference
results = CPUCompiler().log_likelihood(p, inputs)

# Compare with the inference result from SPFlow.
reference = log_likelihood(p, inputs).reshape(30)
assert(np.all(np.isclose(results, reference)))
```

As you can see in the example above, `CPUCompiler().log_likelihood()` can be used as a direct 
replacement of `log_likelihood` from SPFlow, producing equivalent results, but typically much faster.

If you want to compile for CUDA GPUs, just use `from spnc.gpu import CUDACompiler` and 
`GPUCompiler().log_likelihood()` in the code above. Compilation for CUDA GPUs is only available if 
your installation of `spnc` was configured to support CUDA GPUs, you can easily check that through 
`CUDACompiler.isAvailable()` in your Python code.

More details on the usage of the compilers and the available tuning knobs can be found in the 
Python documentation, accessible through `help(CPUCompiler)` and `help(CUDACompiler)`, respectively.

#### Standalone-Usage of xspn ####

The small `xspn` library can also be installed and used independently of the compiler, e.g.,
to persistently serialize SPNs trained with SPFlow in a binary format supporting round-trips.

An SPN graph from SPFlow can be wrapped in a model as follows:

```python
from xspn.structure.Model import SPNModel
spn = [...]
model = SPNModel(spn)
```

A model can further be wrapped in a query:

```python
from xspn.structure.Query import JointProbability
query = JointProbability(model)
```

Finally, the query can be serialized into binary format:

```python
from xspn.serialization.binary.BinarySerialization import BinarySerializer
BinarySerializer("test.bin").serialize_to_file(query)
```

Serialized models can also be de-serialized to Python again:

```python
from xspn.serialization.binary.BinarySerialization import BinaryDeserializer
deserialized_query = BinaryDeserializer("test.bin").deserialize_from_file()
```