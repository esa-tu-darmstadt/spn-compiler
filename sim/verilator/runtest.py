# IMPORTANT: Call this script from project root directory!

import sys
import os
import shutil
import subprocess
from pathlib import PurePath
import numpy as np
from spn_parser import load_spn_2
from spn.algorithms.Inference import likelihood
from spn.structure.leaves.histogram.Histograms import Histogram

from xspn.structure.Model import SPNModel
from xspn.serialization.binary.BinarySerialization import BinarySerializer

from spnc.fpga import FPGACompiler


def test_single_histogram(spn, sim):
  hist: Histogram = spn

  lb = hist.breaks[0]
  ub = hist.breaks[-1]

  for i in range(lb, ub):
    sim.clock()

    print(f'setting input to {i}')
    sim.setInput(np.array([i]))

    sim.step()

    got = sim.getOutput()
    print(f'got output {got}')

    sim.clock()
    sim.step()


def test_spn(spn, sim):
  data = np.zeros((1, 5))
  expected = likelihood(spn, data)

  sim.setInput(np.zeros(5))
  
  for i in range(100):
    sim.clock()
    sim.step()

  print(f'got: {sim.getOutput()} expected: {expected}')


if __name__ == '__main__':
  if len(sys.argv) <= 1:
    print('Usage: test.py <spn path>')
    exit()

  # load spn from spn file
  spn_path = sys.argv[1]
  spn, var_2_index, _, _ = load_spn_2(spn_path)

  # invoke the driver to compile create the verilog sources
  compiler = FPGACompiler(computeInLogSpace=False)
  try:
    kernel = compiler.compile_ll(spn)
    pass
  except:
    pass

  # invoke the Makefile to compile the verilator shared library + python bindings
  shutil.copy(
    'ipxact_core/src/simimpl.cpp',
    './sim/verilator/bindings/'
  )

  env = {
    **os.environ.copy(),
    'VERILATOR_TOP_SOURCE': '$(BASE_DIR)/spn-compiler/ipxact_core/src/spn_body.v',
    'VERILATOR_INCLUDES': '-I$(BASE_DIR)/spn-compiler/ipxact_core/src/',
    'VERILATOR_SIMIMPL': '$(BASE_DIR)/spn-compiler/ipxact_core/src/simimpl.cpp'
  }

  cwd = './sim/verilator/bindings/'

  print('Calling make...')
  subprocess.run('make pyspnsim', shell=True, cwd=cwd, env=env)

  sys.path.append('./sim/verilator/bindings/')
  import pyspnsim

  print('Starting simulation...')
  sim = pyspnsim.PySPNSim()
  sim.init(sys.argv)

  if isinstance(spn, Histogram):
    test_single_histogram(spn, sim)
  else:
    test_spn(spn, sim)

  sim.final()
