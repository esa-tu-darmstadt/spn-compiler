SHELL := /bin/bash


#c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) example.cpp -o example$(python3-config --extension-suffix)
#.PHONY: pybind
#spnsim: src/spn.hpp src/spn.cpp
#	verilator -Wall -shared -std=c++17 --exe --build -fPIC $(python3 -m pybind11 --includes) $^ -o spnsim$(python3-config --extension-suffix)

INC = -I$(BASE_DIR)/pybind11/include/
INC += -I/usr/include/python3.10/
INC += -I$(BASE_DIR)/verilator/include/
INC += -I$(BASE_DIR)/spn-compiler/sim/verilator/obj_dir/

SRC = src/spnsimpy.cpp

SUFFIX := $$(python3-config --extension-suffix)

.PHONY: spnverilator
spnverilator: src/spn.cpp ipxact_core/src/top.sv
	verilator --lib-create spnverilator --cc --build -Iipxact_core/src/ $^

SIM_INC = -I$(BASE_DIR)/verilator/include/

spnsim: $(SRC) spnverilator
	g++ -Wall -shared -std=c++17 -fPIC $(INC) $(SRC) -o spnsim$(SUFFIX) -Lobj_dir -lspnverilator

test: src/test.cpp #spnverilator
	g++ $^ -shared -std=c++17 -fPIC $(INC) -o mytest$(SUFFIX) -Lobj_dir -lspnverilator

clean:
	rm -f spnsim.cpython-310-x86_64-linux-gnu.so
	rm -rf obj_dir
