# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

import numpy as np

import os

from spn.structure.Base import Product, Sum
from spn.structure.leaves.histogram.Histograms import Histogram
from spn.algorithms.Inference import log_likelihood

from xspn.serialization.binary.BinarySerialization import BinaryDeserializer

from spnc.cpu import CPUCompiler

import pytest


@pytest.mark.skipif(not CPUCompiler.isVectorizationSupported(), reason="CPU vectorization not supported")
def test_vector_NIPS5():
    # Locate test resources located in same directory as this script.
    scriptPath = os.path.realpath(os.path.dirname(__file__))

    # Deserialize model
    query = BinaryDeserializer(os.path.join(scriptPath, "NIPS5.bin")).deserialize_from_file()
    spn = query.graph.root

    inputs = np.genfromtxt(os.path.join(scriptPath, "inputdata.txt"), delimiter=";", dtype="int32")
    # Execute the compiled Kernel.
    results = CPUCompiler(computeInLogSpace=True).log_likelihood(spn, inputs, supportMarginal=False)

    # Compute the reference results using the inference from SPFlow.
    reference = np.genfromtxt(os.path.join(scriptPath, "outputdata.txt"), delimiter=";", dtype="float32")
    reference = reference.reshape(10000)

    # Check the computation results against the reference
    # Check in normal space if log-results are not very close to each other.
    assert np.all(np.isclose(results, reference)) or np.all(np.isclose(np.exp(results), np.exp(reference)))
    

if __name__ == "__main__":
    test_vector_NIPS5()
    print("COMPUTATION OK")
