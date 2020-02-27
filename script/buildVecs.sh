#!/bin/bash

#args :
#1. file that contains one spn link on each line
#2. arguments to pass to driver
#3. name for experiment
#4. timeout in seconds

echo "$3 : $2" >> /Users/johannesschulte/Desktop/Uni/MT/benchmarks/experimentLog
timeoutFile="/Users/johannesschulte/Desktop/Uni/MT/benchmarks/${3}_unfinished.txt"
touch $timeoutFile
while IFS= read -r LINE; do
    spnName="$(basename "$(dirname "$LINE")")"
    outputPath="/Users/johannesschulte/Desktop/Uni/MT/benchmarks/irFiles/${3}_${spnName}.bc"
    ts=$(gdate +%s%N)
    gtimeout $4 /Users/johannesschulte/Desktop/Uni/MT/cpp-spn-compiler/optDebLLVM/execute/driver $LINE $outputPath $2
    execTime=$((($(gdate +%s%N) - $ts)/1000000))
    if [ -f $outputPath ]; then
	echo "$3,$spnName,$execTime" >> /Users/johannesschulte/Desktop/Uni/MT/benchmarks/buildTimes
    else
	echo $LINE >> $timeoutFile
    fi
    
done < $1
