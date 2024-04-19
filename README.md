# SPN Compiler #

[![Main Build](https://github.com/esa-tu-darmstadt/spn-compiler/actions/workflows/release-build-linux.yml/badge.svg)](https://github.com/esa-tu-darmstadt/spn-compiler/actions/workflows/release-build-linux.yml)
[![Development Build](https://github.com/esa-tu-darmstadt/spn-compiler/actions/workflows/weekly-build-linux.yml/badge.svg)](https://github.com/esa-tu-darmstadt/spn-compiler/actions/workflows/weekly-build-linux.yml)

## About SPNC ##

**SPNC** is a multi-target compiler for Sum-Product Networks, a class of machine learning models.

Starting with release 0.0.4, **SPNC** is mostly implemented in `C++` and uses
the [LLVM compiler framework](https://llvm.org/)
and [MLIR](https://mlir.llvm.org) for code generation for the different targets.

Currently supported targets are CPUs (all architectures supported by LLVM, vectorization currently limited to X86 (AVX,
AVX2, AVX-512) and ARM Neon) and CUDA GPUs.


### Installation ###

**SPNC** comprises two main parts: `xspn`, a small library to help with the serialization of SPFlow models, and `spnc`,
which is the compiler itself.

The easiest way to install both components is to use the pre-built Python packages (wheels) provided on the SPNC
Github [release page](https://github.com/esa-tu-darmstadt/spn-compiler/releases). While `xspn` is completely
platform-independent, the pre-built wheel for `spnc` only works on Linux platforms. See
the [installation instructions](https://github.com/esa-tu-darmstadt/spn-compiler/wiki/Installation-Manual) for detailed
requirements.

In case you want to use **SPNC** on a different platform or want to build **SPNC** from source, follow
the [installation instructions](https://github.com/esa-tu-darmstadt/spn-compiler/wiki/Installation-Manual) to build
**SPNC** and all its dependencies from source. Currently, `spnc` is based on LLVM release 17.0.6 (tag `llvmorg-17.0.6`).

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

### Build PyBind11 from source

```
cd $BASE_DIR
git clone https://github.com/pybind/pybind11.git
cd pybind11
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$BASE_DIR/pybind11/install \
     -DPYTHON_EXECUTABLE:FILEPATH=/usr/bin/python3 ..
make
make install
```

### Build spdlog from source

```
cd $BASE_DIR
git clone https://github.com/gabime/spdlog.git
cd spdlog
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$BASE_DIR/spdlog/install -DSPDLOG_BUILD_SHARED=ON ..
make
make install
```

### Build Cap'n'Proto from source

```
cd $BASE_DIR
git clone https://github.com/sandstorm-io/capnproto.git
cd capnproto/c++
autoreconf -i
./configure --prefix=$BASE_DIR/capnproto/install
# Make sure to adapt the number of parallel jobs to your machine.
make -j 16
make install
```

### Build MLIR and CIRCT from source

```
cd $BASE_DIR
git clone https://github.com/jschj/circt.git
cd circt
git checkout lower-esi-to-axi-stream
git submodule update --recursive
cd llvm
git checkout 660b3c85c8ea74909de0116bd1dae1b83342cffa     # just in case the

# build MLIR
cd $BASE_DIR/circt/llvm
mkdir build
cd build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang;compiler-rt" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DBUILD_SHARED_LIBS=ON \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DLLVM_ENABLE_RTTI=ON 
ninja
ninja check-mlir

# build CIRCT
cd $BASE_DIR/circt
mkdir build
cd build

# we need capnproto for ESI
PREFIX_PATH=$PREFIX_PATH"$BASE_DIR/capnproto/install;"
cmake -G Ninja .. \
  -DCMAKE_PREFIX_PATH=$PREFIX_PATH \
  -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
  -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DLLVM_ENABLE_RTTI=ON -DESI_COSIM=ON \
  -DLLVM_ENABLE_LLD=ON \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++
ninja
ninja check-circt
ninja check-circt-integration
```

### Build TaPaSCo from source



### Build SPNC from source

```
cd $BASE_DIR
git clone https://github.com/jschj/spn-compiler.git
cd spn-compiler
git checkout feature/circt-ufloat-ops
git submodule update --recursive
mkdir build
cd build

PREFIX_PATH="$BASE_DIR/circt/llvm/build/lib/cmake/llvm;"
PREFIX_PATH=$PREFIX_PATH"$BASE_DIR/circt/llvm/build/lib/cmake/mlir;"

PREFIX_PATH=$PREFIX_PATH"$BASE_DIR/circt/build/lib/cmake/circt;"

PREFIX_PATH=$PREFIX_PATH"$BASE_DIR/pybind11/install/share/cmake/pybind11;"
PREFIX_PATH=$PREFIX_PATH"$BASE_DIR/spdlog/install/lib64/cmake/spdlog;"
PREFIX_PATH=$PREFIX_PATH"$BASE_DIR/capnproto/install"

cmake -DCMAKE_PREFIX_PATH=$PREFIX_PATH\
  -DLLVM_ENABLE_ASSERTIONS=ON\
  -DSPNC_BUILD_DOC=ON\
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON\
  -DBUILD_SHARED_LIBS=ON\
  -DCMAKE_BUILD_TYPE=Debug\
  -DTAPASCO_FPGA_SUPPORT=1\ # omit if TaPaSCo is not required
  ..

make -jN
```