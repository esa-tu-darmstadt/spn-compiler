import numpy as np
from pathlib import Path
import sys

from spn.structure.Base import Product, Sum
from spn.structure.leaves.histogram.Histograms import Histogram
from spn.algorithms.Inference import log_likelihood, likelihood

from spnc.fpga import FPGACompiler
from xspn.spn_parser import load_spn_2


def print_usage():
    print('usage: compile_fpga.py <path to .spn file> <compile/execute>')

if __name__ == "__main__":
  if len(sys.argv) <= 2:
    print_usage()
    exit()

  spn_path = sys.argv[1]
  cmd = sys.argv[2]

  spn, var_2_index, index_to_min, index_to_max = load_spn_2(spn_path)

  wdir_name = Path(spn_path).stem

  if cmd == 'compile':
    compiler = FPGACompiler(computeInLogSpace=False)
    kernel = compiler.compile_normal(spn, Path(wdir_name))
  elif cmd == 'execute':
    compiler = FPGACompiler(computeInLogSpace=False)
    kernel = compiler.compile_kernel_info(spn, Path(wdir_name))

    inputs = np.zeros((8, len(index_to_min.items())))

    print(f'inputs={inputs}')

    num_samples = inputs.shape[0]
    outputs = kernel.execute(num_samples, inputs)
    expected = likelihood(spn, inputs)

    print(np.hstack((expected, outputs)))

    
