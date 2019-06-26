## About `spnc` ##

`spnc` is a multi-target compiler for Sum-Product Networks, 
a class of machine learning models.

`spnc` currently supports three different kinds of code generation for SPNs:
* Serial C++ code, using `-t cpp`.
* OpenMP thread parallel C++ code, using `-t cpp` and `--openmp-parallel` 
in combination.
* CUDA code for Nvidia GPUs, using `-t cuda`.

Support for generating SIMD execution with OpenMP is under way. 
Additionally, `spnc` allows to compute
statistics about an SPNs graph and output them to JSON.

### Prerequisites ### 

`spnc` requires a recent version of Java to run. For compilation and execution 
of C++/OpenMP code, at least one of `g++` or `clang++` needs to be installed on 
your machine. If you want to use CUDA compilation for Nvidia GPUs, you need to 
have `nvcc`.

### Installation ###

Either download one of the release zips from the github page or 
compile `spnc` from source. To compile from source, simply run 
`./gradlew installDist` in the root folder of `spnc` and use the executable 
found at `build/install/spnc/bin/spnc`.

### Usage ###

`spnc` compiles SPNs from a textual representation 
(see `src/main/resources/NIPS5.spn` for an example). 

Running `spnc` with the desired input-file will generate an executable. 
The executable contains an automatically generated `main`-method which will 
read input-data from a plain-text file and, if a second file-name is given,
compare them to the reference data from the reference-file (also plain text).
The executable will also track the time spent to execute the SPN inference 
for all examples in the input sample. This works independent from the actual 
execution mode (C++ serial, OpenMP, CUDA on Nvidia GPU).

Run `spnc --help` to see additional options.

##### Limitations #####
`spnc` *currently does not yet support Poisson or Gaussian distributions.*