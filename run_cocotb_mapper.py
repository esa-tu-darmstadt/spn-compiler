"""
Just execute this script with python3 run_cocotb <path to .spn file> from the root directory.
"""

import sys
from pathlib import Path
import subprocess
import os
import argparse

from spn.io.Text import spn_to_str_ref_graph


def bin_name(spn_name: str):
    return spn_name.split('.')[0] + '.bin'

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run cocotb testbench for the full SPN AXI integration')
  parser.add_argument('--spn', metavar='spn', type=str, help='path to the SPN file')
  parser.add_argument('--wdir', metavar='wdir', type=str, help='name of the working directory')
  parser.add_argument('--json-config', metavar='json_config', type=str, help='path to the json config file (or inline json string)', default='')
  parser.add_argument('--exponent-width', metavar='exponent_width', type=int, help='width of the exponent', default=8)
  parser.add_argument('--mantissa-width', metavar='mantissa_width', type=int, help='width of the mantissa', default=23)
  parser.add_argument('--float-type', metavar='float_type', type=str, help='type of the floating point numbers', default='float32')
  args = parser.parse_args()

  # setup PYTHONPATH
  sys.path.append(str(Path('python-interface').resolve()))
  sys.path.append(str(Path('xspn').resolve()))

  # serialize SPN
  from xspn.spn_parser import load_spn
  from xspn.serialization.binary.BinarySerialization import BinarySerializer
  from xspn.structure.Model import SPNModel
  from xspn.structure.Query import JointProbability, ErrorKind

  spn_path = Path(args.spn)
  spn, variables_to_index, index_to_min, index_to_max = load_spn(str(spn_path))

  # create HDL source files from SPN (invoke compiler)
  from spnc.fpga import FPGACompiler, get_fpga_device_config
  
  wdir = Path(args.wdir)
  fpga = FPGACompiler()
  
  # json_config can be a json string or a file path
  if args.json_config == '':
    json_config = get_fpga_device_config('vc709', 'project_vcd')
  else:
    json_config = args.json_config

  # copy json config to wdir
  #json_config_path = wdir / 'config.json'
  #with open(json_config_path, 'w') as file:
  #  file.write(json_config)

  kernel = fpga.compile_testbench(spn, wdir, json_config, exponent_width=args.exponent_width, mantissa_width=args.mantissa_width, float_type=args.float_type)

  # set environment variables for cocotb that point to the correct location
  debug_info_path = (wdir / 'ipxact_core' / 'debug_info.json').resolve()
  hdl_sources_path = (wdir / 'ipxact_core' / 'src').resolve()

  # call cocotb Makefile
  cmd = f'make -C sim/controller/tests/ INCLUDE_DIR={hdl_sources_path}'
  env = {
    **os.environ.copy(),
    'PYTHONPATH': ':'.join(sys.path),
    'SPN_PATH': str(Path(spn_path).resolve()),
    'CONFIG_PATH': str((wdir / 'ipxact_core' / 'config.json').resolve())
  }
  subprocess.run(cmd, shell=True, cwd='.', env=env)