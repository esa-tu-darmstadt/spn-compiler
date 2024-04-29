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
        os.path.join(scriptPath, "nltcs_100_200_2_10_8_8_1_True.bin")
    ).deserialize_from_file()
    spn = model.root

    inputs = np.genfromtxt(
        os.path.join(scriptPath, "input.csv"), delimiter=",", dtype="float64"
    )
    reference = np.genfromtxt(
        os.path.join(scriptPath, "nltcs_100_200_2_10_8_8_1_True_output.csv"),
        delimiter=",",
        dtype="float64",
    )
    reference = reference.reshape(10000)
    # Compile the kernel.
    compiler = CPUCompiler(spnc_cpu_vectorize=True, spnc_use_log_space=True)
    kernel = compiler.compile_ll(spn=spn, batchSize=1, supportMarginal=False)
    # Execute the compiled Kernel.
    time_sum = 0
    for i in range(len(reference)):
        # Check the computation results against the reference
        start = time.time()
        result = compiler.execute(kernel, inputs=np.array([inputs[i]]))
        time_sum = time_sum + time.time() - start
        if not np.isclose(result, reference[i]):
            print(
                f"\nevaluation #{i} failed: result: {result[0]:16.8f}, reference: {reference[i]:16.8f}"
            )
            raise AssertionError()
    print(f"\nExecution of {len(reference)} samples took {time_sum} seconds.")


if __name__ == "__main__":
    test_vector_fashion_mnist()
    print("COMPUTATION OK")
