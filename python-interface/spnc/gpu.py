""" Option split:

Compiler:
verbose -> dumpIR
computeinLogspace
preloadToSharedMem


compile/log_likelihood:

errorModel
batchSize
supportMarginal

"""

import numpy as np
import tempfile
import os
import glob

from xspn.serialization.binary.BinarySerialization import BinarySerializer
from xspn.structure.Model import SPNModel
from xspn.structure.Query import JointProbability, ErrorModel
import spnc.spncpy as spncpy

from ctypes import CDLL

def convertToFlag(value):
    return "true" if value else "false"

class CUDACompiler:
    __cudaWrappers = None

    """Convenience interface to SPNC, targeting execution on CUDA/Nvidia GPUs.

    Args:
        preloadToSharedMem (bool): Pre-load input values to GPU shared memory before computation.
        computeInLogSpace (bool): Perform computations in log-space.
        verbose (bool): Verbose output.

    Attributes:
        preloadToSharedMem (bool): Pre-load input values to GPU shared memory before computation.
        computeInLogSpace (bool): Perform computations in log-space.
        verbose (bool): Verbose output.
    
    """

    

    def __init__(self, preloadToSharedMem = False, computeInLogSpace = True, verbose = False):
        self.verbose = verbose
        self.preloadToSharedMem = preloadToSharedMem
        self.computeInLogSpace = computeInLogSpace

    def compile_ll(self, spn, inputDataType = "float64", errorModel = ErrorModel(), 
                    batchSize = 64, supportMarginal = True, name = "spn_gpu"):
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

        # The MLIR CUDA runtime wrappers are copied to the same location as 
        # the spncpy module, so we supply this directory as an additional library search path.
        libraryDir = os.path.realpath(os.path.dirname(spncpy.__file__))

        # Compile the query into a Kernel.
        options = dict({"target" : "CUDA", 
                        "gpu-shared-mem" : convertToFlag(self.preloadToSharedMem),
                        "use-log-space" : convertToFlag(self.computeInLogSpace),
                        "dump-ir" : convertToFlag(self.verbose),
                        "search-paths" : str(libraryDir)
                        })
        if self.verbose:
            print(f"Invoking compiler with options: {options}")

        kernel = spncpy.SPNCompiler().compileQuery(tmpfile.name, options)
        # Check that the comiled Kernel actually exists.
        if not os.path.isfile(kernel.fileName()):
            raise RuntimeError("Compilation failed, not Kernel produced")

        return kernel

    def _initializeCUDAWrappers():
        # As the MLIR CUDA runtime wrappers library is typically not available on
        # a default library search path and to avoid requiring the user to explicitly 
        # specify LD_LIBRARY_PATH each time anew, we try to load.
        # The packaging process usually adds the library to the spnc package, into
        # the same location as the spncpy compiled module. 
        # We try to locate the library in this directory and load it.
        libraryDir = os.path.realpath(os.path.dirname(spncpy.__file__))
        pattern = os.path.join(libraryDir, "libcuda-runtime-wrappers.so*")
        candidates = glob.glob(pattern)
        if not candidates:
            print(f"WARNING: Did not find MLIR CUDA runtime wrappers in {libraryDir}")
        lib = candidates[0]
        # Load the shared library to make it available to the compiled GPU kernel.
        CUDACompiler.__cudaWrappers = CDLL(lib)

    def execute(self, kernel, inputs):
        if type(inputs) is not np.ndarray:
            raise RuntimeError("Input is not an numpy array")
        if inputs.ndim != 2:
            raise RuntimeError("Input must be a two-dimensional array")

        if not CUDACompiler.__cudaWrappers:
            CUDACompiler._initializeCUDAWrappers()
        numSamples = inputs.shape[0]
        results = kernel.execute(numSamples, inputs)
        return results

    def log_likelihood(self, spn, inputs, errorModel = ErrorModel(),
                        batchSize = 64, supportMarginal = True):
        if type(inputs) is not np.ndarray:
            raise RuntimeError("Input is not an numpy array")
        if inputs.ndim != 2:
            raise RuntimeError("Input must be a two-dimensional array")

        dataType = inputs.dtype
        kernel = self.compile_ll(spn, str(dataType), errorModel = errorModel, 
                            batchSize = batchSize, supportMarginal = supportMarginal, 
                            name = "spn_gpu")
        results = self.execute(kernel, inputs)
        os.remove(kernel.fileName())
        return results

    @staticmethod
    def isAvailable():
        return spncpy.SPNCompiler.isTargetSupported("CUDA")