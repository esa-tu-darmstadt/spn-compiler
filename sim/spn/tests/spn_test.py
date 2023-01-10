import numpy as np

import cocotb
from cocotb.triggers import Timer, RisingEdge, FallingEdge
from cocotb.clock import Clock


async def reset(dut):
  dut.in_0.value = 0
  dut.in_1.value = 0
  dut.in_2.value = 0
  dut.in_3.value = 0
  dut.in_4.value = 0

  dut.rst.value = 1
  for _ in range(3):
    await RisingEdge(dut.clk)
  dut.rst.value = 0


async def write_inputs(ports, inputs, delay):
  if delay > 0:
    await Timer(delay, units="ns")

  for row in inputs:
    for i in range(5):
      ports[i].value = int(row[i])
    
    await Timer(2, units="ns")


async def read_outputs(port, count, delay):
  await Timer(delay, units="ns")
  results = []

  for i in range(count):
    results.append(port.value)
    await Timer(2, units="ns")

  print(f'GOT:')
  print(results)

async def run_clock(dut):
  for i in range(100):
    print("tick")
    dut.clk.value = 0
    await Timer(1, units="ns")
    dut.clk.value = 1
    await Timer(1, units="ns")
  print("done")


async def generate_clock(dut):
  """Generate clock pulses."""

  for cycle in range(10):
    dut._log.info("tick")
    dut.clk.value = 0
    await Timer(1, units="ns")
    dut.clk.value = 1
    await Timer(1, units="ns")


@cocotb.test()
async def spn_basic_test(dut):
  # setup
  inputs = np.array([
    [0, 0, 0, 0, 0]
  ])
  outputs = np.array([
    1
  ])

  spn_delay = 123

  in_ports = [
    dut.in_0, dut.in_1, dut.in_2, dut.in_3, dut.in_4
  ]

  out_port = dut.out_prob

  #cocotb.start_soon(Clock(dut.clk, 2, units="ns").start())
  await cocotb.start(generate_clock(dut))  # run the clock "in the background"

  await Timer(5, units="ns")  # wait a bit
  await FallingEdge(dut.clk)  # wait for falling edge/"negedge"
  
  dut._log.info("my_signal_1 is %s", dut.my_signal_1.value)
  assert dut.my_signal_2.value[0] == 0, "my_signal_2[0] is not 0!"

  return

  print("resetting...")
  await reset(dut)
  print("done")

  # write inputs and read outputs concurrently
  write_thread = cocotb.start_soon(
    write_inputs(in_ports, inputs, 0)
  )

  read_thread = cocotb.start_soon(
    read_outputs(out_port, outputs.shape[0], spn_delay)
  )

  await write_thread
  await read_thread
  await clk_thread

