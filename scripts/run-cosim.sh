#!/bin/bash

# REQUIRES: esi-cosim
# RUN: rm -rf %t6 && mkdir %t6 && cd %t6
# RUN: circt-opt %s --esi-connect-services --esi-emit-collateral=schema-file=%t2.capnp > %t4.mlir
# RUN: circt-opt %t4.mlir --lower-esi-to-physical --lower-esi-ports --lower-esi-to-hw --export-split-verilog -o %t3.mlir
# RUN: circt-translate %t4.mlir -export-esi-capnp -verify-diagnostics > %t2.capnp
# RUN: cd ..
# RUN: esi-cosim-runner.py --schema %t2.capnp --exec %S/loopback.py %t6/*.sv


rm -rf tmp_cosim
mkdir tmp_cosim
cd tmp_cosim

$BASE_DIR/spn-compiler/build/execute/driver $BASE_DIR/spn-compiler/nips5.bin --target FPGA --controller-generator-path "" || true

circt-opt output.mlir --mlir-disable-threading --esi-connect-services --esi-emit-collateral=schema-file=schema.capnp > x.mlir
#circt-opt x.mlir --lower-esi-to-physical --lower-esi-ports --lower-esi-to-hw --export-split-verilog -o y.mlir
#circt-translate x.mlir -export-esi-capnp -verify-diagnostics > schema.capnp
# RUN: cd ..
# RUN: esi-cosim-runner.py --schema %t2.capnp --exec %S/loopback.py %t6/*.sv