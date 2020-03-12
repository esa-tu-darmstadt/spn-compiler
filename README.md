# SPN Compiler #

## About `spnc` ##

`spnc` is a multi-target compiler for Sum-Product Networks, a class of machine learning models.

As of release 0.0.4, `spnc` is completely implemented in `C++` and uses the LLVM compiler framework
for code generation for the different targets.

## Installation ##

### Prerequisites ###

`spnc` requires a C++ compiler that supports at least the `C++14` standard as well as a 
modern CMake (>= version 3.5)).

### Building from source ###

The following procedure has been tested under Ubuntu 19.10, after installing the packages 
`doxygen graphviz llvm-dev clang`.

`$BASE_DIR` is used as a placeholder for a directory of your choice, replace it in the 
following commands:

First, install and build pybind11:
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

Second, install and build spdlog:
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

Now, download and build `spnc`:
```
cd $BASE_DIR
git clone git@github.com:esa-tu-darmstadt/spn-compiler.git
cd spn-compiler
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="$BASE_DIR/pybind11/install/share/cmake/pybind11;$BASE_DIR/spdlog/install/lib/cmake/spdlog" ..
make
make install
```

## Usage ##

The `execute`-subproject contains a simple `driver` that can be used to run the compiler 
on an SPN serialized to a JSON-file. 

In addition, the compiler also has a pybind11-based Python interface (see `python-interface`).
A demonstration of the interface can be found in the 
[Python examples](https://github.com/esa-tu-darmstadt/spn-examples).