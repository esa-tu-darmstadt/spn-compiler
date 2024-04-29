# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

import numpy as np

from spn.structure.Base import Product, Sum
from spn.structure.leaves.parametric.Parametric import Categorical
from spn.algorithms.Inference import log_likelihood

from spnc.cpu import CPUCompiler

import pytest


@pytest.mark.skipif(
    not CPUCompiler.isVectorizationSupported(), reason="CPU vectorization not supported"
)
def test_log_vector_categorical_no_shuffle():
    # Construct a minimal SPN
    c1 = Categorical(p=[0.35, 0.55, 0.1], scope=0)
    c2 = Categorical(p=[0.25, 0.625, 0.125], scope=1)
    c3 = Categorical(p=[0.5, 0.2, 0.3], scope=2)
    c4 = Categorical(p=[0.6, 0.15, 0.25], scope=3)
    c5 = Categorical(p=[0.7, 0.11, 0.19], scope=4)
    c6 = Categorical(p=[0.8, 0.14, 0.06], scope=5)
    p = Product(children=[c1, c2, c3, c4, c5, c6])

    # Randomly sample input values.
    inputs = np.column_stack(
        (
            np.random.randint(3, size=30),
            np.random.randint(3, size=30),
            np.random.randint(3, size=30),
            np.random.randint(3, size=30),
            np.random.randint(3, size=30),
            np.random.randint(3, size=30),
        )
    ).astype("int32")

    if not CPUCompiler.isVectorizationSupported():
        print("Test not supported by the compiler installation")
        return 0

    # Execute the compiled Kernel.
    results = CPUCompiler(spnc_use_vector_shuffle=False).log_likelihood(
        p, inputs, supportMarginal=False
    )

    # Compute the reference results using the inference from SPFlow.
    reference = log_likelihood(p, inputs)
    reference = reference.reshape(30)

    # Check the computation results against the reference
    # Check in normal space if log-results are not very close to each other.
    assert np.all(np.isclose(results, reference)) or np.all(
        np.isclose(np.exp(results), np.exp(reference))
    )


if __name__ == "__main__":
    test_log_vector_categorical_no_shuffle()
    print("COMPUTATION OK")
