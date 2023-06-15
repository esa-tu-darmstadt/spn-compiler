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
    'S_AXI_LITE_AW_VALID',
    'S_AXI_LITE_AW_ADDR',
    'S_AXI_LITE_AW_PROT',
    'S_AXI_LITE_W_VALID',
    'S_AXI_LITE_W_DATA',
    'S_AXI_LITE_W_STRB',
    'S_AXI_LITE_B_READY',
    'S_AXI_LITE_AR_VALID',
    'S_AXI_LITE_AR_ADDR',
    'S_AXI_LITE_AR_PROT',
    'S_AXI_LITE_R_READY',
    'M_AXI_AW_READY',
    'M_AXI_W_READY',
    'M_AXI_B_VALID',
    'M_AXI_B_ID',
    'M_AXI_B_RESP',
    'M_AXI_B_USER',
    'M_AXI_AR_READY',
    'M_AXI_R_VALID',
    'M_AXI_R_ID',
    'M_AXI_R_DATA',
    'M_AXI_R_RESP',
    'M_AXI_R_USER',
    'M_AXI_R_LAST',
    'S_AXI_LITE_AW_READY',
    'S_AXI_LITE_W_READY',
    'S_AXI_LITE_B_VALID',
    'S_AXI_LITE_B_RESP',
    'S_AXI_LITE_AR_READY',
    'S_AXI_LITE_R_VALID',
    'S_AXI_LITE_R_DATA',
    'S_AXI_LITE_R_RESP',
    'M_AXI_AW_VALID',
    'M_AXI_AW_ID',
    'M_AXI_AW_ADDR',
    'M_AXI_AW_LEN',
    'M_AXI_AW_SIZE',
    'M_AXI_AW_BURST',
    'M_AXI_AW_LOCK',
    'M_AXI_AW_CACHE',
    'M_AXI_AW_PROT',
    'M_AXI_AW_QOS',
    'M_AXI_AW_REGION',
    'M_AXI_AW_USER',
    'M_AXI_W_VALID',
    'M_AXI_W_DATA',
    'M_AXI_W_STRB',
    'M_AXI_W_LAST',
    'M_AXI_B_READY',
    'M_AXI_AR_VALID',
    'M_AXI_AR_ID',
    'M_AXI_AR_ADDR',
    'M_AXI_AR_LEN',
    'M_AXI_AR_SIZE',
    'M_AXI_AR_BURST',
    'M_AXI_AR_LOCK',
    'M_AXI_AR_CACHE',
    'M_AXI_AR_PROT',
    'M_AXI_AR_QOS',
    'M_AXI_AR_REGION',
    'M_AXI_AR_USER',
    'M_AXI_R_READY'
  ]

  for signal in signals:
    parts = signal.split('_')
    #prefix = '_'.join(parts[0:-1])
    #suffix = parts[-1]
    lower = '_'.join(parts[0:-2]) + '_' + parts[-2].lower() + parts[-1].lower()

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
  spn_path = os.environ['SPN_PATH']
  spn, var_2_index, index_2_min, index_2_max = load_spn_2(spn_path)
  COUNT = 20 * 1000
  data = generate_data(COUNT, index_2_min, index_2_max)
  expected = likelihood(spn, data, dtype=np.float32)

  cfg = load_config()

  rename_signals(dut)

  print('=====================INPUTS=====================')
  print(data)

  # assume 1 byte per value
  read_base = 0x10000
  read_size = COUNT * len(index_2_min.items())
  # IPECStoreUnit does not adhere to the standard in the sense that it crosses 4k boundaries
  write_base = roundN(read_base + read_size, 4096)
  write_size = COUNT * (4 if cfg['floatType'] == 'float32' else 8)

  # from tapasco
  #size_t loadBeatCount = inSize / (fpgaKernel.memDataWidth / 8);
  #size_t storeBeatCount = outSize / (fpgaKernel.memDataWidth / 8);
  # TODO: read_size and write_size need to be padded first!
  load_beat_count = read_size // (cfg['axi4']['dataWidth'] // 8)
  store_beat_count = write_size // (cfg['axi4']['dataWidth'] // 8)

  print(f'bytes to send: {read_size} in {load_beat_count} beats on a 4 byte bus')
  print(f'bytes to receive: {write_size} in {store_beat_count} beats on a 4 byte bus')

  #required_space = COUNT * len(index_2_min.items()) + COUNT * 8

  reg_file = AxiLiteMaster(AxiLiteBus.from_prefix(dut, "S_AXI_LITE"), dut.clock, dut.reset)
  axi_ram = AxiRam(AxiBus.from_prefix(dut, "M_AXI"), dut.clock, dut.reset, size=2**32)

  # write the generated input data to RAM
  axi_ram.write(read_base, data.astype(np.int8).tobytes())

  cocotb.fork(Clock(dut.clock, 1, units='us').start())

  print(f'resetting...')
  dut.reset.value = 1
  for i in range(0, 50):
      await RisingEdge(dut.clock)
  #await Timer(5, units="ns")
  dut.reset.value = 0
  print(f'done')

  # write registers
  off = cfg['axi4Lite']['dataWidth'] // 8
  await reg_file.write(2 * off, read_base.to_bytes(off, byteorder='little')) # loadBaseAddress
  await reg_file.write(3 * off, load_beat_count.to_bytes(off, byteorder='little')) # numLdTransfers
  await reg_file.write(4 * off, write_base.to_bytes(off, byteorder='little')) # storeBaseAddress
  await reg_file.write(5 * off, store_beat_count.to_bytes(off, byteorder='little')) # numStTransfers
  await reg_file.write(0 * off, int(1).to_bytes(off, byteorder='little')) # status, okay let's go

  await First(
    cocotb.start_soon(poll_interrupt(dut)),
    cocotb.start_soon(wait_timeout(100000))
  )

  # we didn't crash => we got an interrupt and can now check the results
  as_floats = list(struct.unpack(f'<{write_size // 4}f', axi_ram.read(write_base, write_size)))

  for got, exp in zip(as_floats, expected):
    print(f'got {got} expected {exp}')
    assert math.isclose(got, exp, rel_tol=1e-5), f'got {got} expected {exp}'
