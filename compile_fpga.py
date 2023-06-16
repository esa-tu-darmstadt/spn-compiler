"""
Just execute this script with python3 compile_fpga.py <path to .spn file> from the root directory.
"""

import sys
from pathlib import Path
import subprocess
import os

from spn.io.Text import spn_to_str_ref_graph


def print_usage():
  print('Usage: compile_fpga <compile/load/execute> <path to .spn file> <work directory name>')

def bin_name(spn_name: str):
  return spn_name.split('.')[0] + '.bin'

if __name__ == '__main__':
  if len(sys.argv) <= 3:
    print_usage()
    exit(0)

  # setup PYTHONPATH
  sys.path.append(str(Path('python-interface').resolve()))
  sys.path.append(str(Path('xspn').resolve()))

  print('sys.path:')
  print(sys.path)

  # serialize SPN
  from xspn.spn_parser import load_spn
  from xspn.serialization.binary.BinarySerialization import BinarySerializer
  from xspn.structure.Model import SPNModel
  from xspn.structure.Query import JointProbability, ErrorKind

  cmd = sys.argv[1]
  spn_path = sys.argv[2]
  wdir_name = Path(sys.argv[3])
  spn, variables_to_index, index_to_min, index_to_max = load_spn(str(spn_path))

  # create HDL source files from SPN (invoke compiler)
  from spnc.fpga import FPGACompiler, get_fpga_device_config

  spn_name = Path(spn_path).stem.split('.')[0]

  fpga = FPGACompiler(verbose=False)
  json_config = get_fpga_device_config('vc709')
  
  if cmd == 'compile':
    kernel = fpga.compile_full(spn, wdir_name, json_config)
  elif cmd == 'load':
    pass
  elif cmd == 'execute':
    
    
    
    pass
  else:
    print_usage()
    exit(0)
