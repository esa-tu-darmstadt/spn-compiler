# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

import numpy as np
import os
import pytest
import time
from spnc.cpu import CPUCompiler
from xspn.serialization.binary.BinarySerialization import BinaryDeserializer


@pytest.mark.skipif(not CPUCompiler.isVectorizationSupported(), reason="CPU vectorization not supported")
def test_vector_slp_fashion_mnist():
    # Locate test resources located in same directory as this script.
    scriptPath = os.path.realpath(os.path.dirname(__file__))

    # Deserialize model
    model = BinaryDeserializer(
        os.path.join(scriptPath, "fashion_mnist_200_100_4_5_10_9_1_True.bin")).deserialize_from_file()
    spn = model.root

    inputs = np.genfromtxt(os.path.join(scriptPath, "input.csv"), delimiter=",", dtype="float64")
    reference = np.genfromtxt(os.path.join(scriptPath, "fashion_mnist_200_100_4_5_10_9_1_True_output.csv"),
                              delimiter=",",
                              dtype="float64")
    reference = reference.reshape(1000)

    # Compile the kernel.
    options = {}
    options["slp-max-look-ahead"] = 10
    options["slp-max-node-size"] = 10000
    options["slp-max-attempts"] = 5
    options["slp-max-successful-iterations"] = 1
    options["slp-reorder-dfs"] = True
    options["slp-allow-duplicate-elements"] = False
    options["slp-allow-topological-mixing"] = False
    options["slp-use-xor-chains"] = True

    # Compile the kernel with batch size 1 to enable SLP vectorization.
    compiler = CPUCompiler(vectorize=True, computeInLogSpace=True, vectorLibrary="LIBMVEC", **options)
    kernel = compiler.compile_ll(spn=spn, batchSize=1, supportMarginal=False)
    # Execute the compiled Kernel.
    time_sum = 0
    for i in range(len(reference)):
        # Check the computation results against the reference
        start = time.time()
        result = compiler.execute(kernel, inputs=np.array([inputs[i]]))
        time_sum = time_sum + time.time() - start
        if not np.isclose(result, reference[i]):
            print(f"\nevaluation #{i} failed: result: {result[0]:16.8f}, reference: {reference[i]:16.8f}")
            raise AssertionError()
    print(f"\nExecution of {len(reference)} samples took {time_sum} seconds.")


if __name__ == "__main__":
    test_vector_slp_fashion_mnist()
    print("COMPUTATION OK")
