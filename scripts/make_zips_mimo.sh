#!/bin/bash
JSON_CONFIG='{"device":{"name":"vc709","mhz":200},"axi4":{"addrWidth":32,"dataWidth":512},"axi4Lite":{"addrWidth":32,"dataWidth":32},"kernelId":1}'

# nips
python compile_fpga.py package --spn resources/spns/NIPS5/structure.spn --project-name NIPS5_spnc_1_mimo --json-config $JSON_CONFIG --exponent-width 7 --mantissa-width 24 --float-type float64
#python compile_fpga.py package --spn resources/spns/NIPS10/structure.spn --project-name NIPS10_spnc_1_mimo --json-config $JSON_CONFIG --exponent-width 7 --mantissa-width 24 --float-type float64
#python compile_fpga.py package --spn resources/spns/NIPS20/structure.spn --project-name NIPS20_spnc_1_mimo --json-config $JSON_CONFIG --exponent-width 8 --mantissa-width 24 --float-type float64
#python compile_fpga.py package --spn resources/spns/NIPS30/structure.spn --project-name NIPS30_spnc_1_mimo --json-config $JSON_CONFIG --exponent-width 8 --mantissa-width 26 --float-type float64
#python compile_fpga.py package --spn resources/spns/NIPS40/structure.spn --project-name NIPS40_spnc_1_mimo --json-config $JSON_CONFIG --exponent-width 9 --mantissa-width 26 --float-type float64
#python compile_fpga.py package --spn resources/spns/NIPS50/structure.spn --project-name NIPS50_spnc_1_mimo --json-config $JSON_CONFIG --exponent-width 9 --mantissa-width 26 --float-type float64
#python compile_fpga.py package --spn resources/spns/NIPS60/structure.spn --project-name NIPS60_spnc_1_mimo --json-config $JSON_CONFIG --exponent-width 9 --mantissa-width 26 --float-type float64
#python compile_fpga.py package --spn resources/spns/NIPS70/structure.spn --project-name NIPS70_spnc_1_mimo --json-config $JSON_CONFIG --exponent-width 10 --mantissa-width 26 --float-type float64
#python compile_fpga.py package --spn resources/spns/NIPS80/structure.spn --project-name NIPS80_spnc_1_mimo --json-config $JSON_CONFIG --exponent-width 10 --mantissa-width 27 --float-type float64

# binary networks
#python compile_fpga.py package --spn resources/spns/ACCIDENTS_4000/structure.spn --project-name ACCIDENTS_4000_spnc_1_mimo --json-config $JSON_CONFIG --exponent-width 10 --mantissa-width 26 --float-type float64
#python compile_fpga.py package --spn resources/spns/BAUDIO_4000/structure.spn --project-name BAUDIO_4000_spnc_1_mimo --json-config $JSON_CONFIG --exponent-width 10 --mantissa-width 26 --float-type float64
#python compile_fpga.py package --spn resources/spns/BNETFLIX_4000/structure.spn --project-name BNETFLIX_4000_spnc_1_mimo --json-config $JSON_CONFIG --exponent-width 10 --mantissa-width 26 --float-type float64
#python compile_fpga.py package --spn resources/spns/MSNBC_200/structure.spn --project-name MSNBC_200_spnc_1_mimo --json-config $JSON_CONFIG --exponent-width 10 --mantissa-width 26 --float-type float64
#python compile_fpga.py package --spn resources/spns/MSNBC_300/structure.spn --project-name MSNBC_300_spnc_1_mimo --json-config $JSON_CONFIG --exponent-width 10 --mantissa-width 26 --float-type float64
#python compile_fpga.py package --spn resources/spns/NLTCS_200/structure.spn --project-name NLTCS_200_spnc_1_mimo --json-config $JSON_CONFIG --exponent-width 10 --mantissa-width 26 --float-type float64
#python compile_fpga.py package --spn resources/spns/PLANTS_4000/structure.spn --project-name PLANTS_4000_spnc_1_mimo --json-config $JSON_CONFIG --exponent-width 10 --mantissa-width 26 --float-type float64