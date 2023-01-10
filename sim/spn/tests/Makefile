SIM ?= verilator
TOPLEVEL_LANG ?= verilog

PWD=$(shell pwd)

export PYTHONPATH := $(PWD)/../model:$(PYTHONPATH)

EXTRA_ARGS+=-I$(PWD)/../hdl

VERILOG_SOURCES += $(PWD)/../hdl/top.sv

TOPLEVEL := spn_body
MODULE   := spn_test

include $(shell cocotb-config --makefiles)/Makefile.sim