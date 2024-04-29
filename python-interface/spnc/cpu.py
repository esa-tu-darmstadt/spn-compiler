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


def convertToFlag(value):
    return "true" if value else "false"


class VectorLibrary(enum.Enum):
    Accelerate = "Accelerate"
    Darwin_libsystem_m = "Darwin_libsystem_m"
    LIBMVEC = "LIBMVEC"
    MASSV = "MASSV"
    SVML = "SVML"
    sleefgnuabi = "SLEEF"
    ArmPL = "ArmPL"
    AMDLIBM = "AMDLIBM"
    NoLibrary = "None"
    Default = "Default"


class CPUCompiler:
    """Convenience interface to SPNC, targeting execution on the CPU.

    Attributes
    ----------
    spnc_cpu_vectorize: bool
        Perform vectorization if possible.
    spnc_vector_library : VectorLibrary
        The vector library to use for optimized math functions.
    spnc_use_log_space : bool
        Perform computations in log-space.
    spnc_use_vector_shuffle : bool
        Use vector shuffles instead of gather loads when vectorizing.
    spnc_dump_ir : bool
        spnc_dump_ir output.
    otherOptions : dict
        Additional options to pass to the compiler.

    Methods
    -------

    compile_ll(spn, inputDataType = "float64", errorModel = ErrorModel(),
                    batchSize = 4096, supportMarginal = True, name = "spn_cpu")
        Compile SPN and return the compiled kernel.

    execute(kernel, inputs)
        Execute a previously compiled kernel on the given inputs.

    log_likelihood(spn, inputs, errorModel = ErrorModel(), batchSize = 4096, supportMarginal = True)
        Compile the SPN and immediately execute the compiled kernel on the given inputs.

    isVectorizationSupported()
        Check whether the compiler supports vectorization.


    """

    def __init__(
        self,
        spnc_cpu_vectorize=True,
        spnc_vector_library=VectorLibrary.Default,
        spnc_use_log_space=True,
        spnc_use_vector_shuffle=True,
        spnc_dump_ir=False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        spnc_cpu_vectorize: bool, optional
            Perform vectorization if possible.
        spnc_vector_library : VectorLibrary, optional
            The vector library to use for optimized math functions, possible values "ArmPL",
        spnc_use_log_space : bool, optional
            Perform computations in log-space.
        spnc_use_vector_shuffle : bool, optional
            Use vector shuffles instead of gather loads when vectorizing.
        spnc_dump_ir : bool, optional
            spnc_dump_ir output.
        kwargs :
            Additional options to pass to the compiler.
        """

        self.spnc_dump_ir = spnc_dump_ir
        self.spnc_cpu_vectorize = spnc_cpu_vectorize
        if spnc_vector_library == VectorLibrary.Default:
            self.spnc_vector_library = CPUCompiler.getDefaultVectorLibrary()
        else:
            self.spnc_vector_library = spnc_vector_library
        self.spnc_use_log_space = spnc_use_log_space
        self.spnc_use_vector_shuffle = spnc_use_vector_shuffle
        self.otherOptions = kwargs

    def compile_ll(
        self,
        spn,
        inputDataType="float64",
        errorModel=ErrorModel(),
        batchSize=4096,
        supportMarginal=True,
        name="spn_cpu",
    ):
        """Compile the SPN for the CPU target and return the compiled kernel.

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
                "spnc-target": "CPU",
                "spnc-cpu-vectorize": convertToFlag(self.spnc_cpu_vectorize),
                "spnc-vector-library": self.spnc_vector_library.value,
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
        # Check that the comiled Kernel actually exists.
        if not os.path.isfile(kernel.fileName()):
            raise RuntimeError("Compilation failed, not Kernel produced")

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
        results = kernel.execute(numSamples, inputs)
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
        os.remove(kernel.fileName())
        return results

    @staticmethod
    def isVectorizationSupported():
        """Query the compiler for vectorization support on the host CPU"""

        return spncpy.SPNCompiler.isFeatureAvailable("spnc_cpu_vectorize")

    @staticmethod
    def getDefaultVectorLibrary():
        hostArch = spncpy.SPNCompiler.getHostArchitecture()
        if hostArch == "aarch64":
            return VectorLibrary.ArmPL
        elif hostArch == "x86_64":
            return VectorLibrary.LIBMVEC
        else:
            return VectorLibrary.NoLibrary
