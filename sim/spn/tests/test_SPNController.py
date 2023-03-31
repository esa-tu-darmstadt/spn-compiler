
import numpy as np
import struct
from codecs import decode
import math
import json
import sys
import os

import cocotb
from cocotb.triggers import Timer, RisingEdge, FallingEdge
from cocotb.clock import Clock

from cocotbext.axi import AxiStreamBus, AxiStreamSource, AxiStreamSink, AxiStreamFrame

from xspn.spn_parser import load_spn_2
#from spn_parser import load_spn_2
from spn.algorithms.Inference import likelihood
from spn.structure.leaves.histogram.Histograms import Histogram

from functools import reduce

from init import prepare


def load_config() -> dict:
  path = os.environ['CONFIG_PATH']

  with open(path, 'r') as file:
    content = file.read()
    return json.loads(content)

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

  for i in range(len(data.tdata) // 8):
    b = data.tdata[i * 8 : (i + 1) * 8]
    got.append(struct.unpack('d', b))

  return got

def generate_data(count, index_2_min, index_2_max):
  np.random.seed(123456)
  var_count = len(index_2_min.items())
  data = np.zeros((count, var_count), dtype=np.int32)

  for j in range(var_count):
    low = index_2_min[j]
    high = index_2_max[j]

    for i in range(count):
      val = 0.0 if np.random.uniform() < 0.9 else np.random.randint(low, high)
      data[i][j] = val

  return data

# cocotb does not like capslock signal names Q_Q
def rename_signals(dut):
  signals = ["AXIS_SLAVE_TVALID",
             "AXIS_SLAVE_TDATA",
             "AXIS_SLAVE_TSTRB",
             "AXIS_SLAVE_TKEEP",
             "AXIS_SLAVE_TLAST",
             "AXIS_SLAVE_TUSER",
             "AXIS_SLAVE_TDEST",
             "AXIS_SLAVE_TID",
             "AXIS_MASTER_TREADY",
             "AXIS_SLAVE_TREADY",
             "AXIS_MASTER_TVALID",
             "AXIS_MASTER_TDATA",
             "AXIS_MASTER_TSTRB",
             "AXIS_MASTER_TKEEP",
             "AXIS_MASTER_TLAST",
             "AXIS_MASTER_TUSER",
             "AXIS_MASTER_TDEST",
             "AXIS_MASTER_TID"]

  for signal in signals:
    parts = signal.split('_')
    prefix = parts[0] + '_' + parts[1]
    suffix = parts[-1]
    lower = suffix.lower()

    print(f'Linking signals: {prefix + "_" + lower} <-> {signal}')
    dut.__dict__[prefix + '_' + lower] = dut.__getattr__(signal)

@cocotb.test()
async def test_SPNController(dut):
  spn_path = os.environ['SPN_PATH']
  spn, var_2_index, index_2_min, index_2_max = load_spn_2(spn_path)
  COUNT = 1000
  data = generate_data(COUNT, index_2_min, index_2_max)
  print(f'data.shape={data.shape}')
  expected = likelihood(spn, data, dtype=np.float32)

  rename_signals(dut)

  axis_source = AxiStreamSource(AxiStreamBus.from_prefix(dut, "AXIS_SLAVE"), dut.clock, dut.reset)
  axis_sink = AxiStreamSink(AxiStreamBus.from_prefix(dut, "AXIS_MASTER"), dut.clock, dut.reset)

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

  # TODO: fix case where result = expected = 0
  results = np.array(results).reshape((data.shape[0], 1))
  error = np.abs((results - expected) / expected)
  # verbose = True

  print(f'Max relative error is {np.max(error)}')

  if np.max(error) > 1e-3 or np.any(np.isnan(error)) or verbose:
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
