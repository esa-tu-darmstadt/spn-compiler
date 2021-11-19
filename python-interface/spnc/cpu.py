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

class CPUCompiler:
    """Convenience interface to SPNC, targeting execution on the CPU.

    Attributes
    ----------
    vectorize: bool
        Perform vectorization if possible.
    vectorLibrary : str
        Use vector library for optimized math functions, possible values "ARM", "LIBMVEC", "SVML", "Default" and "None".
    computeInLogSpace : bool
        Perform computations in log-space.
    useVectorShuffle : bool
        Use vector shuffles instead of gather loads when vectorizing.
    verbose : bool
        Verbose output.
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

    def __init__(self, vectorize=True, vectorLibrary="Default", computeInLogSpace=True,
                 useVectorShuffle=True, verbose=False, **kwargs):
        """
        Parameters
        ----------
        vectorize: bool, optional
            Perform vectorization if possible.
        vectorLibrary : str, optional
            Use vector library for optimized math functions, possible values "ARM",
            "LIBMVEC", "SVML", "Default" and "None".
        computeInLogSpace : bool, optional
            Perform computations in log-space.
        useVectorShuffle : bool, optional
            Use vector shuffles instead of gather loads when vectorizing.
        verbose : bool, optional
            Verbose output.
        kwargs :
            Additional options to pass to the compiler.
        """

        self.verbose = verbose
        self.vectorize = vectorize
        if vectorLibrary == "Default":
            self.vectorLibrary = CPUCompiler.getDefaultVectorLibrary()
        else:
            self.vectorLibrary = vectorLibrary
        self.computeInLogSpace = computeInLogSpace
        self.useVectorShuffle = useVectorShuffle
        self.otherOptions = kwargs


    def compile_ll(self, spn, inputDataType = "float64", errorModel = ErrorModel(), 
                    batchSize = 4096, supportMarginal = True, name = "spn_cpu"):
        """ Compile the SPN for the CPU target and return the compiled kernel.

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
        options = dict({"target": "CPU",
                        "cpu-vectorize": convertToFlag(self.vectorize),
                        "vector-library": self.vectorLibrary,
                        "use-shuffle": convertToFlag(self.useVectorShuffle),
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

    def log_likelihood(self, spn, inputs, errorModel = ErrorModel(),
                        batchSize = 4096, supportMarginal = True):
        """ Compile the SPN and immediately execute the compiled kernel on the given inputs.

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

        kernel = self.compile_ll(spn, str(dataType), errorModel = errorModel,
                                 batchSize = batchSize, supportMarginal = supportMarginal,
                                 name="spn_cpu")
        results = self.execute(kernel, inputs)
        os.remove(kernel.fileName())
        return results

    @staticmethod
    def isVectorizationSupported():
        """Query the compiler for vectorization support on the host CPU"""

        return spncpy.SPNCompiler.isFeatureAvailable("vectorize")

    @staticmethod
    def getDefaultVectorLibrary():
        hostArch = spncpy.SPNCompiler.getHostArchitecture()
        if hostArch == "aarch64":
            return "ARM"
        elif hostArch == "x86_64":
            return "LIBMVEC"
        else:
            return "None"
