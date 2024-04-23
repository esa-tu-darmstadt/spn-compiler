SHELL := /bin/bash

#VERILATOR_TOP_SOURCE = $(BASE_DIR)/spn-compiler/ipxact_core/src/spn_body.v
#VERILATOR_INCLUDES = -I$(BASE_DIR)/spn-compiler/ipxact_core/src/

#.PHONY: sim
sim: sim.cpp simimpl.cpp $(VERILATOR_TOP_SOURCE)
	verilator --top-module spn_body --lib-create spnverilator --cc --build $(VERILATOR_INCLUDES) $^

PY_INC = -I$(BASE_DIR)/pybind11/include/
PY_INC += $$(python3-config --includes)
PY_INC += -I$(BASE_DIR)/verilator/include/
PY_INC += -I$(BASE_DIR)/spn-compiler/sim/verilator/bindings/obj_dir/

SUFFIX := $$(python3-config --extension-suffix)

pyspnsim: sim pyspnsim.cpp
	g++ -Wall -shared -std=c++17 -fPIC $(PY_INC) pyspnsim.cpp obj_dir/*.o -o pyspnsim$(SUFFIX)

clean:
	rm -rf obj_dir
	rm -f pyspnsim$(SUFFIX)
