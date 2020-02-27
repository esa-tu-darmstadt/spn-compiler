#!/bin/bash

#args :
#1. file that contains one bitcode link on each line


while IFS= read -r LINE; do
    filename="$(basename $LINE .bc)"
    expName=$(echo $filename | cut -f1 -d_)
    spnName=$(echo $filename | cut -f 2- -d_)
    binaryPath="/Users/johannesschulte/Desktop/Uni/MT/xspn-benchmarks/binary/${spnName}/"
    countPath="/Users/johannesschulte/Desktop/Uni/MT/xspn-benchmarks/count/${spnName}/"
    if [ -d "$binaryPath" ]; then
	inputFile="${binaryPath}inputdata.txt"
	outputFile="${binaryPath}outputdata.txt"
    else
	inputFile="${countPath}inputdata.txt"
	outputFile="${countPath}outputdata.txt"
    fi
    binName="/tmp/${filename}.out"
    /usr/local/Cellar/llvm/9.0.0_1/bin/clang++ -ffast-math -mfma -march=skylake -ffp-contract=fast -O3 $LINE /Users/johannesschulte/Desktop/Uni/MT/cpp-spn-compiler/execute/resources/main.cpp -o $binName
    for run in {1..5}
    do
	execTime=$(${binName} $inputFile $outputFile A)  
	echo "$expName,$spnName,$execTime" >> /Users/johannesschulte/Desktop/Uni/MT/benchmarks/runTimes
    done
done < $1
