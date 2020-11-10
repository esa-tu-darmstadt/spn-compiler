# SPNC #

The project is split into a number of subprojects:

* [common](../spnc-common/html/index.html): Common headers used by multiple subprojects,
e.g., to exchange data between compiler and runtime.

* [compiler](../spnc/html/index.html): The actual compiler.

* **compiler-rt**: Runtime functions included into the generated code by the compiler to 
add functionality.

* **execute**: Simple examples for the invocation of the compiler and the runtime.

* **python-interface**: Pybind11-based Python/C++-interface to interact with the compiler from
Python, in particular SPFlow. 

* [runtime](../spnc-rt/html/index.html): Runtime to load and execute generated kernels for
computation. 

* [mlir](../mlir-doc/html/index.html): MLIR dialect and libraries used by the compiler.