# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

import numpy as np
import os
import time
from spnc.cpu import CPUCompiler
from xspn.serialization.binary.BinarySerialization import BinaryDeserializer


def test_vector_fashion_mnist():
    if not CPUCompiler.isVectorizationSupported():
        print("Test not supported by the compiler installation")
        return 0
    # Locate test resources located in same directory as this script.
    scriptPath = os.path.realpath(os.path.dirname(__file__))

    # Deserialize model
    model = BinaryDeserializer(
        os.path.join(scriptPath, "fashion_mnist_200_100_1_14_15_10_1_True.bin")).deserialize_from_file()
    spn = model.root

    inputs = np.genfromtxt(os.path.join(scriptPath, "input.csv"), delimiter=",", dtype="float64")
    reference = np.genfromtxt(os.path.join(scriptPath, "fashion_mnist_200_100_1_14_15_10_1_True_output.csv"),
                              delimiter=",",
                              dtype="float64")
    reference = reference.reshape(1000)
    # Compile the kernel.
    compiler = CPUCompiler(vectorize=False, computeInLogSpace=False)
    kernel = compiler.compile_ll(spn=spn, batchSize=1, supportMarginal=False)
    # Execute the compiled Kernel.
    start = time.time()
    for i in range(len(reference)):
        # Check the computation results against the reference
        result = compiler.execute(kernel, inputs=np.array([inputs[i]]))
        assert np.isclose(result, reference[i])
    end = time.time()
    print(f"Execution took {end - start} seconds.")


if __name__ == "__main__":
    test_vector_fashion_mnist()
    print("COMPUTATION OK")
