#!/bin/bash

JSON_CONFIG='{"device":{"name":"vc709","mhz":200},"axi4":{"addrWidth":32,"dataWidth":512},"axi4Lite":{"addrWidth":32,"dataWidth":32},"projectName":"MyProject","kernelId":1,"floatType":"float32"}'

python run_cocotb_mapper.py --spn resources/spns/NIPS5/structure.spn --wdir NIPS5_sim_1 --json-config $JSON_CONFIG --exponent-width 8 --mantissa-width 23 --float-type float32
python run_cocotb_mapper.py --spn resources/spns/NIPS10/structure.spn --wdir NIPS10_sim_1 --json-config $JSON_CONFIG --exponent-width 8 --mantissa-width 23 --float-type float32
python run_cocotb_mapper.py --spn resources/spns/NIPS10/structure.spn --wdir NIPS10_sim_1 --json-config $JSON_CONFIG --exponent-width 8 --mantissa-width 23 --float-type float32