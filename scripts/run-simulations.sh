#!/bin/bash

JSON_CONFIG='{"device":{"name":"vc709","mhz":200},"axi4":{"addrWidth":32,"dataWidth":512},"axi4Lite":{"addrWidth":32,"dataWidth":32},"kernelId":1}'

#python run_cocotb_mapper.py --spn resources/spns/ACCIDENTS_4000/structure.spn --wdir ACCIDENTS_4000_sim_1 --json-config $JSON_CONFIG --exponent-width 10 --mantissa-width 26 --float-type float64

python run_cocotb_mapper.py --spn resources/spns/NIPS5/structure.spn --wdir NIPS5_sim_1 --json-config $JSON_CONFIG --exponent-width 7 --mantissa-width 24 --float-type float64
#python run_cocotb_mapper.py --spn resources/spns/NIPS10/structure.spn --wdir NIPS10_sim_1 --json-config $JSON_CONFIG --exponent-width 7 --mantissa-width 24 --float-type float64
#python run_cocotb_mapper.py --spn resources/spns/NIPS20/structure.spn --wdir NIPS20_sim_1 --json-config $JSON_CONFIG --exponent-width 8 --mantissa-width 24 --float-type float64
#python run_cocotb_mapper.py --spn resources/spns/NIPS30/structure.spn --wdir NIPS30_sim_1 --json-config $JSON_CONFIG --exponent-width 8 --mantissa-width 26 --float-type float64
#python run_cocotb_mapper.py --spn resources/spns/NIPS40/structure.spn --wdir NIPS40_sim_1 --json-config $JSON_CONFIG --exponent-width 9 --mantissa-width 26 --float-type float64
#python run_cocotb_mapper.py --spn resources/spns/NIPS50/structure.spn --wdir NIPS50_sim_1 --json-config $JSON_CONFIG --exponent-width 9 --mantissa-width 26 --float-type float64

#python run_cocotb_mapper.py --spn resources/spns/NIPS5/structure.spn --wdir NIPS5_sim_1 --json-config $JSON_CONFIG --exponent-width 8 --mantissa-width 23 --float-type float32
#python run_cocotb_mapper.py --spn resources/spns/NIPS10/structure.spn --wdir NIPS10_sim_1 --json-config $JSON_CONFIG --exponent-width 8 --mantissa-width 23
