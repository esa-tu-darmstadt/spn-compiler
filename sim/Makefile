
headers: counter.sv
	verilator --cc $^

all: headers sim_main.cpp
	verilator --cc --exe --build -Wall sim_main.cpp counter.sv
