import numpy as np
import struct
from codecs import decode

import cocotb
from cocotb.triggers import Timer, RisingEdge, FallingEdge
from cocotb.clock import Clock


# https://stackoverflow.com/questions/8751653/how-to-convert-a-binary-string-into-a-float-value/8762541
def int_to_bytes(n, length):  # Helper function
    """ Int/long to byte string.

        Python 3.2+ has a built-in int.to_bytes() method that could be used
        instead, but the following works in earlier versions including 2.x.
    """
    return decode('%%0%dx' % (length << 1) % n, 'hex')[-length:]


def bin_to_float(b):
    """ Convert binary string to a float. """
    bf = int_to_bytes(int(b, 2), 8)  # 8 bytes needed for IEEE 754 binary64.
    return struct.unpack('>d', bf)[0]


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
  for result in results:
    bit_str = str(result)
    f = bin_to_float(bit_str)
    print(f'{bit_str}: {f}')

async def run_clock(dut):
  for i in range(1000):
    print("tick")
    dut.clk.value = 0
    await Timer(1, units="ns")
    dut.clk.value = 1
    await Timer(1, units="ns")
  print("done")


async def generate_clock(dut):
  """Generate clock pulses."""

  for cycle in range(200):
    #dut._log.info("tick")
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

  spn_delay = 50

  in_ports = [
    dut.in_0, dut.in_1, dut.in_2, dut.in_3, dut.in_4
  ]

  out_port = dut.out_prob

  await cocotb.start(generate_clock(dut))  # run the clock "in the background"
  
  # reset internal state
  await reset(dut)

  # write inputs
  await cocotb.start(
    write_inputs(in_ports, inputs, 0)
  )

  # read outputs
  await cocotb.start(
    read_outputs(out_port, 10, spn_delay)
  )

  # wait until everything is done
  await Timer(500, units="ns")  # wait a bit
  await FallingEdge(dut.clk)  # wait for falling edge/"negedge"
  