# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

import numpy as np
import tempfile
import os
import json

from xspn.serialization.binary.BinarySerialization import BinarySerializer
from xspn.structure.Model import SPNModel
from xspn.structure.Query import JointProbability, ErrorModel
import spnc.spncpy as spncpy


def get_fpga_device_config(device_name: str, project_name: str) -> str:
  d = {}

  if device_name == 'vc709':
    d = {
      "device": {
        "name": "vc709",
        "mhz": 200
      },
      "axi4": {
        "addrWidth": 32,
        "dataWidth": 512
      },
      "axi4Lite": {
        "addrWidth": 32,
        "dataWidth": 32
      },
      "projectName": project_name,
      "kernelId": 1
    }
  elif device_name == 'ultra96v2':
    # TODO: check if this is correct
    d = {
      "device": {
        "name": "ultra96v2",
        "mhz": 200
      },
      "axi4": {
        "addrWidth": 64,
        "dataWidth": 64
      },
      "axi4Lite": {
        "addrWidth": 64,
        "dataWidth": 64
      },
      "projectName": project_name,
      "kernelId": 123
    }
  else:
    raise ValueError(f'unknown device {device_name}')

  return json.dumps(d)

def convertToFlag(value):
    return "true" if value else "false"

class FPGACompiler:
    def __init__(self, verbose=False, **kwargs):
        self.verbose = verbose
        self.otherOptions = kwargs

    def load_bitstream(self, bitstream_path: str):
        pass

    def _compile(self, spn, wdir, options, json_config):
        model = SPNModel(spn, 'uint8')
        query = JointProbability(model)

        print(f'using working directory path {wdir}')
        try:
            os.mkdir(wdir)
        except:
            pass
        bin_path = wdir / 'spn.bin'

        # Serialize the SPN to binary format as input to the compiler.
        if self.verbose:
            print(f"Serializing SPN to {bin_path}")
        BinarySerializer(bin_path).serialize_to_file(query)
        # Check that the serialization worked.
        if not os.path.isfile(bin_path):
            raise RuntimeError("Serialization of the SPN failed")

        # Add the extra options, if they do not clash with an existing option.
        if self.otherOptions is not None:
            extraOptions = [(str(k), str(v)) for k, v in self.otherOptions.items()]
            for k, v in extraOptions:
                if k in options and options[k] != v:
                    print(f"WARNING: Option {k} specified twice, ignoring option value {v}")
                else:
                    options[k] = v

        if self.verbose:
            print(f"Invoking compiler with options: {options}")

        print(options)
        # This step will fail and is expected.
        kernel = spncpy.SPNCompiler().compileQuery(str(bin_path), options)
        return kernel

    def compile_testbench(self, spn, wdir, json_config, exponent_width=8, mantissa_width=23, float_type='float32'):
        options = dict({"target": "FPGA",
                        "o": str(wdir),
                        "fpga-wrap-axi-stream": "true",
                        "fpga-create-verilog-files": "true",
                        "fpga-coco-tb": "true",
                        "fpga-config-json": json_config,
                        "fpga-exponent-width": str(exponent_width),
                        "fpga-mantissa-width": str(mantissa_width),
                        "fpga-float-type": float_type
                        })

        return self._compile(spn, wdir, options, json_config)

    def compile_full(self, spn, wdir, json_config, project_name, exponent_width=8, mantissa_width=23, float_type='float32'):
        options = dict({"target": "FPGA",
                        "o": str(wdir),
                        "fpga-wrap-axi-stream": "true",
                        "fpga-create-verilog-files": "true",
                        "fpga-coco-tb": "false",
                        "fpga-config-json": json_config,
                        "vivado": "true",
                        "tapasco-compose": "true",
                        "fpga-exponent-width": str(exponent_width),
                        "fpga-mantissa-width": str(mantissa_width),
                        "fpga-float-type": float_type,
                        "project-name": project_name
                        })

        return self._compile(spn, wdir, options, json_config)

    def compile_get_kernel_info(self, spn, wdir, json_config):
        options = dict({"target": "FPGA",
                        "o": str(wdir),
                        "just-get-kernel": "true",
                        "fpga-config-json": json_config
                        })

        return self._compile(spn, wdir, options, None)

    def execute(self, kernel, inputs):
        if type(inputs) is not np.ndarray:
            raise RuntimeError("Input is not an numpy array")
        if inputs.ndim != 2:
            raise RuntimeError("Input must be a two-dimensional array")
        
        numSamples = inputs.shape[0]
        results = kernel.execute(numSamples, inputs)
        return results
