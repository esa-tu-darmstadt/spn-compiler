"""
Just execute this script with python3 compile_fpga.py <path to .spn file> from the root directory.
"""

import sys
from pathlib import Path
import subprocess
import os
import argparse
import numpy as np

from spn.io.Text import spn_to_str_ref_graph


def print_usage():
  print('Usage: compile_fpga <compile/load/execute> <path to .spn file> <project name>')

def bin_name(spn_name: str):
  return spn_name.split('.')[0] + '.bin'

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run cocotb testbench for the full SPN AXI integration')
  parser.add_argument('cmd', metavar='cmd', type=str, help='command to execute (compile/load/execute)')
  parser.add_argument('--spn', metavar='spn', type=str, help='path to the SPN file')
  parser.add_argument('--project-name', metavar='project_name', type=str, help='name of the project and directory')
  parser.add_argument('--json-config', metavar='json_config', type=str, help='path to the json config file (or inline json string)', default='')
  parser.add_argument('--exponent-width', metavar='exponent_width', type=int, help='width of the exponent', default=8)
  parser.add_argument('--mantissa-width', metavar='mantissa_width', type=int, help='width of the mantissa', default=23)
  parser.add_argument('--float-type', metavar='float_type', type=str, help='type of the floating point numbers', default='float32')
  args = parser.parse_args()

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

  cmd = args.cmd
  spn_path = args.spn
  project_name = args.project_name
  wdir_name = Path(project_name)
  spn, variables_to_index, index_to_min, index_to_max = load_spn(str(spn_path))

  # create HDL source files from SPN (invoke compiler)
  from spnc.fpga import FPGACompiler, get_fpga_device_config

  fpga = FPGACompiler(verbose=False)
  json_config = get_fpga_device_config('vc709', project_name) if args.json_config == '' else args.json_config

  if cmd == 'compile':
    kernel = fpga.compile_full(spn, wdir_name, json_config, project_name, args.exponent_width, args.mantissa_width, args.float_type)
  elif cmd == 'load':
    pass
  elif cmd == 'execute':
    pass
    var_count = len(variables_to_index)
    COUNT = 20
    input_data = np.zeros((COUNT, var_count), dtype=np.uint8)

    kernel = fpga.compile_get_kernel_info(spn, wdir_name, json_config)
    fpga.execute(kernel, input_data)
  elif cmd == 'test':
    pass
    kernel = fpga.compile_full(spn, wdir_name, json_config)
  else:
    print('Unknown command: ' + cmd)
    exit(1)
