# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

import enum
import numpy as np
import tempfile
import os

from xspn.serialization.binary.BinarySerialization import BinarySerializer
from xspn.structure.Model import SPNModel
from xspn.structure.Query import JointProbability, ErrorModel
import spnc.spncpy as spncpy
import spnc.spncpy.ipu as ipu_rt


def convertToFlag(value):
    return "true" if value else "false"


class IPUCompiler:
    """Convenience interface to SPNC, targeting execution on the IPU."""

    def __init__(
        self,
        spnc_ipu_target="Model",
        spnc_max_task_size=5,
        spnc_use_log_space=True,
        spnc_use_vector_shuffle=True,
        spnc_dump_ir=False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        spnc_ipu_target : str, optional
            Target to compile for: "Model", "IPU1", "IPU2" or "IPU21".
        spnc_max_task_size : int, optional
            Maximum number of nodes per task.
        spnc_use_log_space : bool, optional
            Perform computations in log-space.
        spnc_use_vector_shuffle : bool, optional
            Use vector shuffles instead of gather loads when vectorizing.
        spnc_dump_ir : bool, optional
            spnc_dump_ir output.
        kwargs :
            Additional options to pass to the compiler.
        """
        self.spnc_ipu_target = spnc_ipu_target
        self.spnc_max_task_size = spnc_max_task_size
        self.spnc_dump_ir = spnc_dump_ir
        self.spnc_use_log_space = spnc_use_log_space
        self.spnc_use_vector_shuffle = spnc_use_vector_shuffle
        self.otherOptions = kwargs

    def compile_ll(
        self,
        spn,
        inputDataType="float32",
        errorModel=ErrorModel(),
        batchSize=4096,
        supportMarginal=True,
        name="spn_ipu",
    ):
        """Compile the SPN for the IPU target and return the compiled kernel.

        Parameters
        ----------

        spn : spn.structure.Base.Node
            Root node of the SPN.
        inputDataType : str, optional
            dtype of the input data.
        errorModel : xspn.structure.Query.ErrorModel, optional
            Error requirements
        batchSize : int, optional
            Batch size to optimize for, 1 for single execution
        supportMarginal : bool, optional
            Support marginalized evaluation in compiled kernel
        name : str, optional
            Name of the compiled kernel function.
        """

        model = SPNModel(spn, inputDataType, name)
        query = JointProbability(
            model,
            batchSize=batchSize,
            supportMarginal=supportMarginal,
            rootError=errorModel,
        )

        # Serialize the SPN to binary format as input to the compiler.
        tmpfile = tempfile.NamedTemporaryFile()
        if self.spnc_dump_ir:
            print(f"Serializing SPN to {tmpfile}")
        BinarySerializer(tmpfile.name).serialize_to_file(query)
        # Check that the serialization worked.
        if not os.path.isfile(tmpfile.name):
            raise RuntimeError("Serialization of the SPN failed")

        # Compile the query into a Kernel.
        options = dict(
            {
                "spnc-target": "IPU",
                "spnc-compute-type-width": "32",
                "spnc-log-compute-type-width": "32",
                "spnc-max-task-size": str(self.spnc_max_task_size),
                "spnc-ipu-target": self.spnc_ipu_target,
                "spnc-use-shuffle": convertToFlag(self.spnc_use_vector_shuffle),
                "spnc-use-log-space": convertToFlag(self.spnc_use_log_space),
                "spnc-dump-ir": convertToFlag(self.spnc_dump_ir),
            }
        )

        # Add the extra options, if they do not clash with an existing option.
        if self.otherOptions is not None:
            extraOptions = [(str(k), str(v)) for k, v in self.otherOptions.items()]
            for k, v in extraOptions:
                # Replace "_" with "-" in the option name because "-" is not allowed in Python variable names,
                # but is typically used in the compiler options.
                k = k.replace("_", "-")
                if k in options and options[k] != v:
                    print(
                        f"WARNING: Option {k} specified twice, ignoring option value {v}"
                    )
                else:
                    options[k] = v

        # Append

        if self.spnc_dump_ir:
            print(f"Invoking compiler with options: {options}")

        kernel = spncpy.SPNCompiler().compileQuery(tmpfile.name, options)

        return kernel

    def execute(self, kernel, inputs):
        """Execute a compiled kernel on the given inputs.

        Parameters
        ----------

        kernel : spnc.spncpy.Kernel
            A previously compiled kernel
        inputs : numpy.ndarray
            Input data.
        """

        if type(inputs) is not np.ndarray:
            raise RuntimeError("Input is not an numpy array")
        if inputs.ndim != 2:
            raise RuntimeError("Input must be a two-dimensional array")
        numSamples = inputs.shape[0]

        runtime = ipu_rt.IPURuntime()
        runtime.load(kernel)

        results = runtime.execute(numSamples, inputs)

        runtime.unload()

        return results

    def log_likelihood(
        self, spn, inputs, errorModel=ErrorModel(), batchSize=4096, supportMarginal=True
    ):
        """Compile the SPN and immediately execute the compiled kernel on the given inputs.

        Parameters
        ----------

        spn : spn.structure.Base.Node
            Root node of the SPN.
        inputs : numpy.ndarray
            Input data.
        errorModel : xspn.structure.Query.ErrorModel, optional
            Error requirements
        batchSize : int, optional
            Batch size to optimize for, 1 for single execution
        supportMarginal : bool, optional
            Support marginalized evaluation in compiled kernel
        """

        if type(inputs) is not np.ndarray:
            raise RuntimeError("Input is not an numpy array")
        if inputs.ndim != 2:
            raise RuntimeError("Input must be a two-dimensional array")

        dataType = inputs.dtype

        kernel = self.compile_ll(
            spn,
            str(dataType),
            errorModel=errorModel,
            batchSize=batchSize,
            supportMarginal=supportMarginal,
            name="spn_cpu",
        )
        results = self.execute(kernel, inputs)
        return results
