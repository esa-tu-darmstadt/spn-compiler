import numpy as np

import os

from spn.structure.Base import Product, Sum
from spn.structure.leaves.histogram.Histograms import Histogram
from spn.algorithms.Inference import log_likelihood

from xspn.serialization.binary.BinarySerialization import BinaryDeserializer

from spnc.cpu import CPUCompiler

def test_vector_NIPS5():
    if not CPUCompiler.isVectorizationSupported():
        print("Test not supported by the compiler installation")
        return 0
    # Locate test resources located in same directory as this script.
    scriptPath = os.path.realpath(os.path.dirname(__file__))

    # Deserialize model
    query = BinaryDeserializer(os.path.join(scriptPath, "NIPS5.bin")).deserialize_from_file()
    spn = query.graph.root

    inputs = np.genfromtxt(os.path.join(scriptPath, "inputdata.txt"), delimiter=";", dtype="int32")
    # Execute the compiled Kernel.
    results = CPUCompiler(computeInLogSpace=False).log_likelihood(spn, inputs, supportMarginal=False)

    # Compute the reference results using the inference from SPFlow.
    reference = np.genfromtxt(os.path.join(scriptPath, "outputdata.txt"), delimiter=";", dtype="float64")
    reference = reference.reshape(10000)

    assert(np.all(np.isclose(results, reference)))
    

if __name__ == "__main__":
    test_vector_NIPS5()
    print("COMPUTATION OK")
