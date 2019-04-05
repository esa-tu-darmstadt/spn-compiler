## About `spnc` ##

`spnc` is a multi-target compiler for Sum-Product Networks, 
a class of machine learning models.

`spnc` currently supports generation of serial C++ code for SPNs, 
support for generating multi-threaded OpenMP code and CUDA code for 
Nvidia GPUs is under way. Additionally, `spnc` allows to compute
statistics about an SPNs graph and output them to JSON.

### Prerequisites ### 

`spnc` requires a recent version of Java to run and 
at least one of `g++` or `clang++` to be installed on your machine. 

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
for all examples in the input sample.

Run `spnc --help` to see additional options.

##### Limitations #####
`spnc` *currently does not yet support Poisson or Gaussian distributions.*