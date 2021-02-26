# SPN Compiler #

## About `spnc` ##

`spnc` is a multi-target compiler for Sum-Product Networks, a class of machine learning models.

As of release 0.0.4, `spnc` is completely implemented in `C++` and uses the [LLVM compiler framework](https://llvm.org/) 
and [MLIR](https://mlir.llvm.org) for code generation for the different targets.


## C++-based compiler ##

The actual compiler is implemented in C++ and builds on LLVM & MLIR. 

### Installation ###

#### Prerequisites ####

`spnc` requires a C++ compiler that supports at least the `C++14` standard as well as a 
modern CMake (>= version 3.7)).

`spnc` and its dependencies require a number of libraries & tools to build. On Ubuntu 20.04, these can be installed with:

`apt install -y git gcc clang cmake ninja-build zlib1g zlib1g-dev python3 lld doxygen graphviz autoconf automake libtool`

#### Building from source ####

The following procedure has been tested on Ubuntu 20.04. 
`$BASE_DIR` is used as a placeholder for a directory of your choice, replace it in the 
following commands or make it available via `export BASE_DIR=[...]`:

First, we will build LLVM and its subproject MLIR that are heavily used by the compiler: 

```
cd $BASE_DIR
mkdir llvm
cd llvm
git clone https://github.com/llvm/llvm-project.git llvm-src
cd llvm-src
# Check out specific commit. Other versions of LLVM/MLIR might work, but the following commit ID has been tested
git checkout f8d3f47e1fd09392aa30df83849b25acd8c59a25
cd ..
mkdir llvm-bin
cd llvm-bin

# Make sure to adapt LLVM_PARALLEL_COMPILE_JOBS and LLVM_PARALLEL_LINK_JOBS to your machine's RAM and CPU.
cmake -G Ninja -DLLVM_ENABLE_PROJECTS="mlir;clang;compiler-rt"\
        -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU"\
        -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON\
        -DLLVM_ENABLE_LLD=ON\
        -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_RTTI=ON\
        -DLLVM_PARALLEL_COMPILE_JOBS=16 -DLLVM_PARALLEL_LINK_JOBS=3\
        -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++\
        -DLLVM_OPTIMIZED_TABLEGEN=ON\
        ../llvm-src/llvm
        
# Build LLVM & MLIR. This step might take some time, depending on your machine.
ninja
```

Next, install and build pybind11 for the Python-to-C++ interface:
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

Install and build spdlog for logging in the compiler:
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

Install and build [capnproto](https://capnproto.org) for binary serialization of SPN graphs:

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

Now, download and build `spnc`:
```
cd $BASE_DIR
git clone git@github.com:esa-tu-darmstadt/spn-compiler.git
cd spn-compiler
git checkout develop
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="$BASE_DIR/llvm/llvm-bin/lib/cmake/llvm;$BASE_DIR/llvm/llvm-bin/lib/cmake/mlir;$BASE_DIR/pybind11/install/share/cmake/pybind11;$BASE_DIR/spdlog/install/lib/cmake/spdlog/;$BASE_DIR/capnproto/install/"\
    -DSPNC_BUILD_DOC=ON\
    -DPYTHON_EXECUTABLE:FILEPATH=/usr/bin/python3.8\
    -DCHECK_SPNC_VERBOSE=ON\
    -DBUILD_SHARED_LIBS=ON\
    -DLLVM_ENABLE_LLD=ON\
    -DLLVM_ENABLE_ASSERTIONS=ON\
    ..

# Make sure to adapt the number of parallel jobs to your machine.
make -j 16    

# spnc comes with some llvm-lit based tests. To verify your installation, you
# can execute those after bringing FileCheck to the PATH as demonstrated here:
export PATH=$BASE_DIR/llvm/llvm-bin/bin:$PATH
make check-spnc-mlir
```

### Usage ###

The `execute`-subproject contains a simple `driver` that can be used to run the compiler 
on an SPN serialized to binary format.

An example invocation from `$BASE_DIR/spn-compiler` might look as follows:
`./build/execute/driver execute/examples/mini-example.bin --target CPU`

## Python library ##

`spnc` comes with a small Python library that interacts with [SPFlow](https://spflow.github.io/SPFlow/) 
and can be used to serialize SPNs to a binary format and provide additional information about the model 
& query to the compiler.

### Installation ###

The Python library can be installed using `pip`:

```
cd $BASE_DIR/spn-compiler/xspn
pip install .
# Alternatively, the library can be installed in editable mode to directly reflect 
# changes in your Python environment or venv:
pip install -e .
# You can also run the tests for the library:
python setup.py pytest
```

### Usage ###

An SPN graph from SPFlow can be wrapped in a model as follows:

```
from xspn.structure.Model import SPNModel
spn = [...]
model = SPNModel(spn)
```

A model can further be wrapped in a query:

```
from xspn.structure.Query import JointProbability
query = JointProbability(model)
```

Finally, the query can be serialized for usage with `spnc`:

```
from xspn.serialization.binary.BinarySerialization import BinarySerializer
BinarySerializer("test.bin").serialize_to_file(query)
```

Serialized models can also be de-serialized to Python again:

```
from xspn.serialization.binary.BinarySerialization import BinaryDeserializer
deserialized_query = BinaryDeserializer("test.bin").deserialize_from_file()
```