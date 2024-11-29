# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

import numpy as np

from spn.structure.Base import Product, Sum
from spn.structure.leaves.parametric.Parametric import Gaussian
from spn.algorithms.Inference import log_likelihood

from xspn.serialization.binary.BinarySerialization import BinaryDeserializer
import os

from spnc.ipu import IPUCompiler


def test_ipu_gaussian():
    # Construct a minimal SPN using two Gaussian leaves.
    g1 = Gaussian(mean=0.5, stdev=1, scope=0)
    g2 = Gaussian(mean=0.125, stdev=0.25, scope=1)
    g3 = Gaussian(mean=0.345, stdev=0.24, scope=2)
    g4 = Gaussian(mean=0.456, stdev=0.1, scope=3)
    g5 = Gaussian(mean=0.94, stdev=0.48, scope=4)
    g6 = Gaussian(mean=0.56, stdev=0.42, scope=5)
    g7 = Gaussian(mean=0.76, stdev=0.14, scope=6)
    g8 = Gaussian(mean=0.32, stdev=0.8, scope=7)
    g9 = Gaussian(mean=0.58, stdev=0.9, scope=8)
    g10 = Gaussian(mean=0.14, stdev=0.2, scope=9)
    p = Product(children=[g1, g2, g3, g4, g5, g6, g7, g8, g9, g10])

    # Locate test resources located in same directory as this script.
    scriptPath = os.path.realpath(os.path.dirname(__file__))

    # Read the trained SPN from file
    model = BinaryDeserializer(
        os.path.join(scriptPath, "../examples/speaker/speaker_FADG0.bin")
    ).deserialize_from_file()
    spn = model.graph.root

    # Randomly sample input values from the two Gaussian (normal) distributions.
    inputs = np.column_stack(
        (
            np.random.normal(0.5, 1, 30),
            np.random.normal(0.125, 0.25, 30),
            np.random.normal(0.345, 0.24, 30),
            np.random.normal(0.456, 0.1, 30),
            np.random.normal(0.94, 0.48, 30),
            np.random.normal(0.56, 0.42, 30),
            np.random.normal(0.76, 0.14, 30),
            np.random.normal(0.32, 0.8, 30),
            np.random.normal(0.58, 0.9, 30),
            np.random.normal(0.14, 0.2, 30),
        )
    ).astype("float32")

    # Execute the compiled Kernel.
    results = IPUCompiler(
        spnc_ipu_target="Model",
        spnc_max_task_size=1000,
        spnc_use_log_space=False,
        spnc_cpu_vectorize=False,
        spnc_dump_ir=False,
    ).log_likelihood(spn, inputs, supportMarginal=False, batchSize=10)

    # Compute the reference results using the inference from SPFlow.
    reference = log_likelihood(p, inputs)
    reference = reference.reshape(30)

    # Check the computation results against the reference
    # Check in normal space if log-results are not very close to each other.
    assert np.all(np.isclose(results, reference)) or np.all(
        np.isclose(np.exp(results), np.exp(reference))
    )


if __name__ == "__main__":
    test_ipu_gaussian()
    print("COMPUTATION OK")
