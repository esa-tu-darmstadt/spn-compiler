"""
Just execute this script with python3 run_cocotb <path to .spn file> from the root directory.
"""

import sys
from pathlib import Path
import subprocess
import os

from spn.io.Text import spn_to_str_ref_graph


def print_usage():
  print('Usage: run_cocotb_mapper <path to .spn file>')

def bin_name(spn_name: str):
    return spn_name.split('.')[0] + '.bin'

if __name__ == '__main__':
  if len(sys.argv) <= 1:
    print_usage()
    exit(0)

  # setup PYTHONPATH
  sys.path.append(str(Path('python-interface').resolve()))
  sys.path.append(str(Path('xspn').resolve()))

  # serialize SPN
  from xspn.spn_parser import load_spn
  from xspn.serialization.binary.BinarySerialization import BinarySerializer
  from xspn.structure.Model import SPNModel
  from xspn.structure.Query import JointProbability, ErrorKind

  spn_path = sys.argv[1]
  spn, variables_to_index, index_to_min, index_to_max = load_spn(str(spn_path))

  print(f'got {spn_to_str_ref_graph(spn)}')

  # create HDL source files from SPN (invoke compiler)
  from spnc.fpga import FPGACompiler
  
  wdir = Path('./spn_core_wdir')
  fpga = FPGACompiler()
  json_config = 'resources/config/vc709-example.json'
  kernel = fpga.compile_testbench(spn, wdir, json_config)

  # set environment variables for cocotb that point to the correct location
  debug_info_path = (wdir / 'ipxact_core' / 'debug_info.json').resolve()
  hdl_sources_path = (wdir / 'ipxact_core' / 'src').resolve()

  # call cocotb Makefile
  cmd = f'make -C sim/controller/tests/ INCLUDE_DIR={hdl_sources_path}'
  env = {
    **os.environ.copy(),
    'PYTHONPATH': ':'.join(sys.path),
    'SPN_PATH': str(Path(spn_path).resolve()),
    'CONFIG_PATH': str(Path(json_config).resolve())
  }
  subprocess.run(cmd, shell=True, cwd='.', env=env)