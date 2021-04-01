""" Option split:

Compiler:
verbose -> dumpIR
vectorize
vectorLibrary
computeinLogspace
shuffle


compile/log_likelihood:

errorModel
batchSize
supportMarginal

"""

import numpy as np
import tempfile
import os

from xspn.serialization.binary.BinarySerialization import BinarySerializer
from xspn.structure.Model import SPNModel
from xspn.structure.Query import JointProbability, ErrorModel
import spncpy as spnc

def convertToFlag(value):
    return "true" if value else "false"

class CPUCompiler:
    """Convenience interface to SPNC, targeting execution on the CPU.

    Args:
        vectorize (bool): Perform vectorization if possible.
        vectorLibrary (str): Use vector library for optimized math functions, 
                            possible values "LIBMVEC", "SVML" and "None".
        computeInLogSpace (bool): Perform computations in log-space.
        useVectorShuffle (bool): Use vector shuffles instead of gather loads when vectorizing.
        verbose (bool): Verbose output.

    Attributes:
        vectorize (bool): Perform vectorization if possible.
        vectorLibrary (str): Use vector library for optimized math functions, 
                            possible values "LIBMVEC", "SVML" and "None".
        computeInLogSpace (bool): Perform computations in log-space.
        useVectorShuffle (bool): Use vector shuffles instead of gather loads when vectorizing.
        verbose (bool): Verbose output.
    
    """

    def __init__(self, vectorize = True, vectorLibrary = "LIBMVEC", computeInLogSpace = True, 
                useVectorShuffle = True, verbose = False):
        self.verbose = verbose
        self.vectorize = vectorize
        self.vectorLibrary = vectorLibrary
        self.computeInLogSpace = computeInLogSpace
        self.useVectorShuffle = useVectorShuffle


    def compile_ll(self, spn, inputDataType = "float64", errorModel = ErrorModel(), 
                    batchSize = 4096, supportMarginal = True, name = "spn_cpu"):
        model = SPNModel(spn, inputDataType, name)
        query = JointProbability(model, batchSize = batchSize, supportMarginal = supportMarginal,
                                 rootError = errorModel)

        # Serialize the SPN to binary format as input to the compiler.
        tempfile = tempfile.NamedTemporaryFile()
        if self.verbose:
            print(f"Serializing SPN to {tempfile}")
        BinarySerializer(tempfile).serialize_to_file(query)
        # Check that the serialization worked.
        if not os.path.isfile(tempfile):
            raise RuntimeError("Serialization of the SPN failed")

        # Compile the query into a Kernel.
        options = dict({"target" : "CPU", 
                        "cpu-vectorize" : convertToFlag(self.vectorize),
                        "vector-library" : self.vectorLibrary,
                        "use-shuffle" : convertToFlag(self.useVectorShuffle),
                        "use-log-space" : convertToFlag(self.computeInLogSpace),
                        "dump-ir" : convertToFlag(self.verbose)
                        })
        if self.verbose:
            print(f"Invoking compiler with options: {options}")

        kernel = spncpy.SPNCompiler().compileQuery(tempfile, options)
        # Check that the comiled Kernel actually exists.
        if not os.path.isfile(kernel.fileName()):
            os.remove(tempfile)
            raise RuntimeError("Compilation failed, not Kernel produced")

        os.remove(tempfile)
        return kernel

    def execute(self, kernel, inputs):
        if type(inputs) is not np.ndarray:
            raise RuntimeError("Input is not an numpy array")
        if inputs.ndim != 2:
            raise RuntimeError("Input must be a two-dimensional array")
        numSamples = inputs.shape[0]
        results = kernel.execute(numSamples, inputs)
        return inputs

    def log_likelihood(self, spn, inputs, errorModel = ErrorModel(),
                        batchSize = 4096, supportMarginal = True, name = "spn_cpu"):
        if type(inputs) is not np.ndarray:
            raise RuntimeError("Input is not an numpy array")
        if inputs.ndim != 2:
            raise RuntimeError("Input must be a two-dimensional array")

        dataType = inputs.dtype

        kernel = compile_ll(self, spn, dataType, errorModel = errorModel, 
                            batchSize = batchSize, supportMarginal = supportMarginal, 
                            name = name)
        results = execute(self, kernel, inputs)
        os.remove(kernel.filename())
        return results
