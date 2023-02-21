
import numpy as np
import struct
from codecs import decode
import math

import cocotb
from cocotb.triggers import Timer, RisingEdge, FallingEdge
from cocotb.clock import Clock

from cocotbext.axi import AxiStreamBus, AxiStreamSource, AxiStreamSink, AxiStreamFrame

from spn_parser import load_spn_2
from spn.algorithms.Inference import likelihood
from spn.structure.leaves.histogram.Histograms import Histogram

from functools import reduce

from init import prepare


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
  # every element gehts interpreted as a byte
  to_send = []

  for row in data:
    for el in row:
      to_send.append(int(el))

  # cocotb splits this up into the appropriate number of frames
  await axis_source.send(to_send)

async def read_outputs(axis_sink):
  data = await axis_sink.recv()

  got = []

  for i in range(len(data.tdata) // 4):
    b = data.tdata[i * 4 : (i + 1) * 4]
    got.append(struct.unpack('f', b))

  return got

def generate_data(count, index_2_min, index_2_max):
  np.random.seed(123456)
  var_count = len(index_2_min.items())
  data = np.zeros((count, var_count), dtype=np.int32)

  for j in range(var_count):
    low = index_2_min[j]
    high = index_2_max[j]

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
async def test_SPNController(dut):
  #prepare(True, 'SPNController')

  spn_path = '../../../examples/nips5.spn'
  spn, var_2_index, index_2_min, index_2_max = load_spn_2(spn_path)
  COUNT = 100000
  data = generate_data(COUNT, index_2_min, index_2_max)
  expected = likelihood(spn, data)

  rename_signals(dut)

  axis_source = AxiStreamSource(AxiStreamBus.from_prefix(dut, "AXI_SLAVE"), dut.clock, dut.reset)
  axis_sink = AxiStreamSink(AxiStreamBus.from_prefix(dut, "AXI_MASTER"), dut.clock, dut.reset)
  cocotb.fork(Clock(dut.clock, 1, units='ns').start())

  print(f'source width: {len(axis_source.bus.tdata)}')
  #print(f'sink width: {len(axis_sink.bus.tdata)}')

  print(f'resetting...')
  dut.reset <= 1
  for i in range(0, 5):
      await RisingEdge(dut.clock)
  dut.reset <= 0
  print(f'done')

  # write inputs
  await write_inputs(axis_source, data)

  # read outputs
  results = await read_outputs(axis_sink)

  # wait until everything is done
  await Timer(1000, units="ns")  # wait a bit
  await FallingEdge(dut.clock)  # wait for falling edge/"negedge"
  
  expected = likelihood(spn, data)
  results = np.array(results).reshape((data.shape[0], 1))
  error = np.abs((results - expected) / expected)

  print(f'Max relative error is {np.max(error)}')

  if np.max(error) > 1e-3:
    for e, r in zip(expected, results):
      print(f'exp={e} got={r}')

  assert np.max(error) <= 1e-3

"""
`ifdef COCOTB_SIM
  initial begin
    $dumpfile("SPNController.vcd");
    $dumpvars (0, SPNController);
    #1;
  end
`endif
"""