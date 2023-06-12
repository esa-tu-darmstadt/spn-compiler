SHELL=/bin/bash
#SIM = questa
SIM = verilator
TOPLEVEL_LANG ?= verilog

PWD=$(shell pwd)

#export PYTHONPATH=$(PWD)/../model:$(PYTHONPATH)
#export PYTHONPATH=$(shell "$$PYTHONPATH"):$(PYS)

#EXTRA_ARGS+=-I$(PWD)/../hdl

# Not my idea: https://github.com/7FM/Usb20DeviceController/blob/master/USBController/Makefile
rwildcard=$(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))

INCLUDE_DIR ?= $(PWD)/../hdl/

VERILOG_SOURCES := $(call rwildcard,$(INCLUDE_DIR),*.v)
VERILOG_SOURCES += $(call rwildcard,$(INCLUDE_DIR),*.sv)

TOPLEVEL := AXI4CocoTbTop
MODULE   := test_AXI4CocoTbTop

VSIM_ARGS=-t 1us

include $(shell cocotb-config --makefiles)/Makefile.sim