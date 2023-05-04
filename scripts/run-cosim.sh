#!/bin/bash

rm -rf tmp_cosim
mkdir tmp_cosim
cd tmp_cosim

$BASE_DIR/spn-compiler/build/execute/driver $BASE_DIR/spn-compiler/nips5.bin --target FPGA --controller-generator-path "" || true

cp $BASE_DIR/spn-compiler/resources/ufloat/FPLog.v FPLog.sv
cp $BASE_DIR/spn-compiler/resources/ufloat/FPOps_build_add/FPAdd.v FPAdd.sv
cp $BASE_DIR/spn-compiler/resources/ufloat/FPOps_build_convert/FP2DoubleConverter.v FP2DoubleConverter.sv
cp $BASE_DIR/spn-compiler/resources/ufloat/FPOps_build_mult/FPMult.v FPMult.sv

#circt-opt output.mlir --mlir-disable-threading --esi-connect-services --esi-emit-collateral=schema-file=schema.capnp > x.mlir

circt-opt output.mlir --esi-emit-collateral=schema-file=schema.capnp > x.mlir
circt-opt x.mlir --lower-esi-to-physical --lower-esi-ports --lower-esi-to-hw --export-split-verilog -o y.mlir
circt-translate x.mlir -export-esi-capnp -verify-diagnostics > schema.capnp

# we need this, otherwise esi-cosim-runner.py:202 will fail
export PYTHONPATH=""
esi-cosim-runner.py --schema $BASE_DIR/spn-compiler/build/tmp_cosim/schema.capnp --exec $BASE_DIR/spn-compiler/scripts/cosim.py $BASE_DIR/spn-compiler/build/tmp_cosim/*.sv

cat $BASE_DIR/spn-compiler/build/tmp_cosim/cosim.py.d/*.log
