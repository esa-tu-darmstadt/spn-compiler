# IMPORTANT: Call this script from project root directory!

import sys
import os
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
  assert isinstance(spn, Histogram)
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


if __name__ == '__main__':
  if len(sys.argv) <= 1:
    print('Usage: test.py <spn path>')
    exit()

  # load spn from spn file
  spn_path = sys.argv[1]
  spn, _, _, _ = load_spn_2(spn_path)

  # invoke the driver to compile create the verilog sources
  compiler = FPGACompiler(computeInLogSpace=False)
  try:
    compiler.compile_ll(spn)
    pass
  except:
    pass

  # invoke the Makefile to compile the verilator shared library + python bindings
  env = {
    **os.environ.copy(),
    'VERILATOR_TOP_SOURCE': '$(BASE_DIR)/spn-compiler/ipxact_core/src/spn_body.v',
    'VERILATOR_INCLUDES': '-I$(BASE_DIR)/spn-compiler/ipxact_core/src/'
  }

  cwd = './sim/verilator/bindings/'

  print('Calling make...')
  subprocess.run('make pyspnsim', shell=True, cwd=cwd, env=env)

  sys.path.append('./sim/verilator/bindings/')
  import pyspnsim

  print('Starting simulation...')
  sim = pyspnsim.PySPNSim()
  sim.init(sys.argv)

  test_single_histogram(spn, sim)

  sim.final()
