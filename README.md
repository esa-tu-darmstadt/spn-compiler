# SPN Compiler for FPGAs

## Prerequisites

- A working TaPaSCo build of `release/2022.1` and correctly sourced into `PATH`.
- Vivado available in the current `PATH` with version `2021.1`.
- A sourced Python environment with all the installed packages from `SPNC` and `CocoTb` with the AXI extension.

Some workarounds might be required because the currently installed Java version can cause issues with the TaPaSCo build. To override the Java version, set `JAVA_HOME` and export `JAVA_HOME` to `PATH`.

## Disclaimers



## Building

We assume that the current work directory is the root directory containing CIRCT, spn-compiler and all the other parts.

```
export BASE_DIR=$(pwd)
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