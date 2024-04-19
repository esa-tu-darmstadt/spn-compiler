#!/bin/bash
JSON_CONFIG='{"device":{"name":"vc709","mhz":200},"axi4":{"addrWidth":32,"dataWidth":512},"axi4Lite":{"addrWidth":32,"dataWidth":32},"kernelId":1}'

python compile_fpga.py compile --spn resources/spns/ACCIDENTS_4000/structure.spn --project-name ACCIDENTS_4000_synth_1_mimo --json-config $JSON_CONFIG --exponent-width 10 --mantissa-width 26 --float-type float64
