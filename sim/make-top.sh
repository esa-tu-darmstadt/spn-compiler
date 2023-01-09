#!/bin/bash
./build/bin/spnc-opt --convert-lospn-to-fpga --lower-seq-firrtl-to-sv --export-verilog $1 -o /dev/null &> sim/top.sv