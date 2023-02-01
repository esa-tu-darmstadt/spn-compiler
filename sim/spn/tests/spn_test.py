
import numpy as np
import struct
from codecs import decode
import math

import cocotb
from cocotb.triggers import Timer, RisingEdge, FallingEdge
from cocotb.clock import Clock

from cocotbext.axi import AxiStreamBus, AxiStreamSource, AxiStreamSink

from spn_parser import load_spn_2
from spn.algorithms.Inference import likelihood
from spn.structure.leaves.histogram.Histograms import Histogram

from functools import reduce


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


async def write_inputs(axis_source, data):
  def to_bitstring(arr):
    return [int(bit) for bit in ''.join([format(el, '#010b')[2:] for el in arr])]

  for row in data:
    bits = to_bitstring(row)

    print(f'{row} <-> {bits}')

    assert(len(bits) == 5 * 8)
    await axis_source.send(bits)

async def read_outputs(axis_sink, count):
  results = []

  for i in range(count):
    data = await axis_sink.recv()
    got = struct.unpack('f', data.tdata[0:4])[0]
    results.append(got)

  return results

async def generate_clock(dut):
  """Generate clock pulses."""

  for cycle in range(200):
    #dut._log.info("tick")
    dut.clock.value = 0
    await Timer(1, units="ns")
    dut.clock.value = 1
    await Timer(1, units="ns")

def generate_data(count, index_2_min, index_2_max):
  np.random.seed(123456)
  var_count = len(index_2_min.items())
  data = np.zeros((count, var_count), dtype=np.int32)

  for j in range(var_count):
    low = index_2_min[j]
    high = index_2_max[j] + 1

    for i in range(count):
      val = np.random.randint(low, high)
      data[i][j] = val

  return data

# cocotb does not like capslock signal names Q_Q
def rename_signals(dut):
  signals = ["AXI_SLAVE_TVALID",
             "AXI_SLAVE_TDATA",
             "AXI_SLAVE_TSTRB",
             "AXI_SLAVE_TKEEP",
             "AXI_SLAVE_TLAST",
             "AXI_SLAVE_TUSER",
             "AXI_SLAVE_TDEST",
             "AXI_SLAVE_TID",
             "AXI_MASTER_TREADY",
             "AXI_SLAVE_TREADY",
             "AXI_MASTER_TVALID",
             "AXI_MASTER_TDATA",
             "AXI_MASTER_TSTRB",
             "AXI_MASTER_TKEEP",
             "AXI_MASTER_TLAST",
             "AXI_MASTER_TUSER",
             "AXI_MASTER_TDEST",
             "AXI_MASTER_TID"]

  for signal in signals:
    parts = signal.split('_')
    prefix = parts[0] + '_' + parts[1]
    suffix = parts[-1]
    lower = suffix.lower()

    print(f'Linking signals: {prefix + "_" + lower} <-> {signal}')
    dut.__dict__[prefix + '_' + lower] = dut.__getattr__(signal)

@cocotb.test()
async def spn_basic_test(dut):
  spn_path = '../../../examples/nips5.spn'
  spn, var_2_index, index_2_min, index_2_max = load_spn_2(spn_path)
  data = generate_data(100, index_2_min, index_2_max)
  expected = likelihood(spn, data)

  rename_signals(dut)

  axis_source = AxiStreamSource(AxiStreamBus.from_prefix(dut, "AXI_SLAVE"), dut.clock, dut.reset)
  axis_sink = AxiStreamSink(AxiStreamBus.from_prefix(dut, "AXI_MASTER"), dut.clock, dut.reset)
  cocotb.fork(Clock(dut.clock, 1, units='ns').start())

  print(f'source width: {len(axis_source.bus.tdata)}')
  print(f'sink width: {len(axis_sink.bus.tdata)}')

  dut.reset <= 1
  for i in range(0, 5):
      await RisingEdge(dut.clock)
  dut.reset <= 0

  # write inputs
  await write_inputs(axis_source, data)

  # read outputs
  results = await read_outputs(axis_sink, data.shape[0])

  # wait until everything is done
  await Timer(5000, units="ns")  # wait a bit
  await FallingEdge(dut.clock)  # wait for falling edge/"negedge"
  
  #expected = likelihood(spn, np.zeros((100, 5)))
  results = np.array(results).reshape((100, 1))
  print(results.shape)
  print(expected.shape)
  print(np.hstack((results, expected)))