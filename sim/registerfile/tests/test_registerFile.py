import numpy as np
import struct
from codecs import decode
import math
import json
import sys
import os
import binascii

import cocotb
from cocotb.triggers import Timer, RisingEdge, FallingEdge, First
from cocotb.clock import Clock

from cocotbext.axi import (
  AxiStreamBus, AxiStreamSource, AxiStreamSink, AxiStreamFrame,
  AxiBus, AxiMaster, AxiSlave, AxiRam,
  AxiLiteBus, AxiLiteMaster, AxiLiteRam
)

from xspn.spn_parser import load_spn_2
#from spn_parser import load_spn_2
from spn.algorithms.Inference import likelihood
from spn.structure.leaves.histogram.Histograms import Histogram

from functools import reduce

#from remote_pdb import RemotePdb

#rpdb = RemotePdb('127.0.0.1', 4000)


def load_config() -> dict:
  path = os.environ['CONFIG_PATH']

  with open(path, 'r') as file:
    content = file.read()
    return json.loads(content)

  assert False, 'Could not load config'


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
    #b = data.tdata[i * 8 : (i + 1) * 8]
    #got.append(struct.unpack('d', b))
    b = data.tdata[i * 4 : (i + 1) * 4]
    print('got binary: ' + binascii.hexlify(b).decode())
    #b = data.tdata[i * 4 : (i + 1) * 4]
    got.append(struct.unpack('f', b))

  return got

def generate_data(count, index_2_min, index_2_max):
  # [ 0  0 34  0  0]
  # [36  0  0  0  0] causes errors

  np.random.seed(123456)
  var_count = len(index_2_min.items())
  data = np.zeros((count, var_count), dtype=np.int32)

  
  for i in range(count):
    for j in range(var_count):  
      low = index_2_min[j]
      high = index_2_max[j]
      val = 0 if np.random.uniform() < 0.9 else np.random.randint(low, high)
      data[i][j] = val

  return data

# cocotb does not like capslock signal names Q_Q
def rename_signals(dut):
  signals = [
    'S_AXI_LITE_AWVALID',
    'S_AXI_LITE_AWADDR',
    'S_AXI_LITE_AWPROT',
    'S_AXI_LITE_WVALID',
    'S_AXI_LITE_WDATA',
    'S_AXI_LITE_WSTRB',
    'S_AXI_LITE_BREADY',
    'S_AXI_LITE_ARVALID',
    'S_AXI_LITE_ARADDR',
    'S_AXI_LITE_ARPROT',
    'S_AXI_LITE_RREADY',
    'M_AXI_AWREADY',
    'M_AXI_WREADY',
    'M_AXI_BVALID',
    'M_AXI_BID',
    'M_AXI_BRESP',
    'M_AXI_BUSER',
    'M_AXI_ARREADY',
    'M_AXI_RVALID',
    'M_AXI_RID',
    'M_AXI_RDATA',
    'M_AXI_RRESP',
    'M_AXI_RUSER',
    'M_AXI_RLAST',
    'S_AXI_LITE_AWREADY',
    'S_AXI_LITE_WREADY',
    'S_AXI_LITE_BVALID',
    'S_AXI_LITE_BRESP',
    'S_AXI_LITE_ARREADY',
    'S_AXI_LITE_RVALID',
    'S_AXI_LITE_RDATA',
    'S_AXI_LITE_RRESP',
    'M_AXI_AWVALID',
    'M_AXI_AWID',
    'M_AXI_AWADDR',
    'M_AXI_AWLEN',
    'M_AXI_AWSIZE',
    'M_AXI_AWBURST',
    'M_AXI_AWLOCK',
    'M_AXI_AWCACHE',
    'M_AXI_AWPROT',
    'M_AXI_AWQOS',
    'M_AXI_AWREGION',
    'M_AXI_AWUSER',
    'M_AXI_WVALID',
    'M_AXI_WDATA',
    'M_AXI_WSTRB',
    'M_AXI_WLAST',
    'M_AXI_BREADY',
    'M_AXI_ARVALID',
    'M_AXI_ARID',
    'M_AXI_ARADDR',
    'M_AXI_ARLEN',
    'M_AXI_ARSIZE',
    'M_AXI_ARBURST',
    'M_AXI_ARLOCK',
    'M_AXI_ARCACHE',
    'M_AXI_ARPROT',
    'M_AXI_ARQOS',
    'M_AXI_ARREGION',
    'M_AXI_ARUSER',
    'M_AXI_RREADY'
  ]

  for signal in signals:
    parts = signal.split('_')
    #prefix = '_'.join(parts[0:-1])
    #suffix = parts[-1]
    lower = '_'.join(parts[0:-1]) + '_' + parts[-1].lower()

    print(f'Linking signals: {lower} (new) <-> (old) {signal}')
    dut.__dict__[lower] = dut.__getattr__(signal)

async def poll_interrupt(dut):
  await RisingEdge(dut.interrupt)
  print(f'Interrupt!')

async def wait_timeout(n):
  await Timer(n, units="us")
  assert False, 'Timeout!'

def roundN(n, m):
  # round n to the next multiple of m
  return n + (m - n) % m

@cocotb.test()
async def test_AXI4CocoTbTop(dut):
  cfg = load_config()

  reg_file = AxiLiteMaster(AxiLiteBus.from_prefix(dut, "S_AXI_LITE"), dut.clock, dut.reset)

  cocotb.fork(Clock(dut.clock, 1, units='us').start())

  print(f'resetting...')
  dut.reset.value = 1
  for i in range(0, 50):
      await RisingEdge(dut.clock)
  dut.reset.value = 0
  print(f'done')

  cocotb.start_soon(wait_timeout(1000000))

  byteCount = cfg['axi4Lite']['dataWidth'] // 8

  for i in range(10):
    await reg_file.write(0x20, int(4).to_bytes(byteCount, byteorder='little'))
    await reg_file.write(0x30, int(5).to_bytes(byteCount, byteorder='little'))

    await reg_file.write(0x0, int(1).to_bytes(byteCount, byteorder='little'))

    await RisingEdge(dut.interrupt)

    c = await reg_file.read(0x40, byteCount)
    await reg_file.read(0x40, byteCount)
    await reg_file.read(0x40, byteCount)
    await reg_file.read(0x40, byteCount)
    await reg_file.read(0x40, byteCount)
    await reg_file.read(0x40, byteCount)
    await reg_file.read(0x40, byteCount)
    print(f'got {c}')
