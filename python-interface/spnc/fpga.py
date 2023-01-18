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

from xspn.serialization.binary.BinarySerialization import BinarySerializer
from xspn.structure.Model import SPNModel
from xspn.structure.Query import JointProbability, ErrorModel
import spnc.spncpy as spncpy


def convertToFlag(value):
    return "true" if value else "false"

class FPGACompiler:
    def __init__(self, computeInLogSpace=True, verbose=False, **kwargs):
        self.verbose = verbose
        self.computeInLogSpace = computeInLogSpace
        self.otherOptions = kwargs

    def compile_ll(self, spn, inputDataType = "float64", errorModel = ErrorModel(), 
                    batchSize = 4096, supportMarginal = True, name = "spn_cpu"):
        model = SPNModel(spn, inputDataType, name)
        query = JointProbability(model, batchSize = batchSize, supportMarginal = supportMarginal,
                                 rootError = errorModel)

        # Serialize the SPN to binary format as input to the compiler.
        tmpfile = tempfile.NamedTemporaryFile()
        if self.verbose:
            print(f"Serializing SPN to {tmpfile}")
        BinarySerializer(tmpfile.name).serialize_to_file(query)
        # Check that the serialization worked.
        if not os.path.isfile(tmpfile.name):
            raise RuntimeError("Serialization of the SPN failed")

        # Compile the query into a Kernel.
        options = dict({"target": "FPGA",
                        "use-log-space": convertToFlag(self.computeInLogSpace),
                        "dump-ir": convertToFlag(self.verbose)
                        })

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

        # This step will fail and is expected.
        spncpy.SPNCompiler().compileQuery(tmpfile.name, options)
        return None