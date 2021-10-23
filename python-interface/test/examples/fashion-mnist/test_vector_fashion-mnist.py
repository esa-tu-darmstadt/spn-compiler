# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

import numpy as np
import os
import shutil
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
    options = dict()
    options["slp-max-look-ahead"] = 10
    options["slp-max-node-size"] = 10000
    options["slp-max-attempts"] = 5
    options["slp-max-successful-iterations"] = 1
    options["slp-reorder-dfs"] = True
    options["slp-allow-duplicate-elements"] = False
    options["slp-allow-topological-mixing"] = False
    options["slp-use-xor-chains"] = True
    compiler = CPUCompiler(vectorize=True, computeInLogSpace=True, vectorLibrary="LIBMVEC", **options)
    kernel = compiler.compile_ll(spn=spn, batchSize=1, supportMarginal=False)
    shutil.copyfile(kernel.fileName(), os.path.join(scriptPath, "fashion.so"))
    # Execute the compiled Kernel.
    time_sum = 0
    for i in range(len(reference)):
        # Check the computation results against the reference
        start = time.time()
        result = compiler.execute(kernel, inputs=np.array([inputs[i]]))
        time_sum = time_sum + time.time() - start
        print(f"evaluation #{i}: result: {result[0]:16.8f}, reference: {reference[i]:16.8f}", end='\r')
        if not np.isclose(result, reference[i]):
            print(f"\nevaluation #{i} failed: result: {result[0]:16.8f}, reference: {reference[i]:16.8f}")
            raise RuntimeError()
    print(f"\nExecution of {len(reference)} samples took {time_sum} seconds.")


if __name__ == "__main__":
    test_vector_fashion_mnist()
    print("COMPUTATION OK")
