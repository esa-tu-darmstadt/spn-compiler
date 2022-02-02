# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

import numpy as np

from spn.structure.Base import Product, Sum
from spn.structure.leaves.histogram.Histograms import Histogram
from spn.structure.leaves.parametric.Parametric import Categorical
from spn.algorithms.Inference import log_likelihood

from spnc.cpu import CPUCompiler

import pytest


@pytest.mark.skipif(not CPUCompiler.isVectorizationSupported(), reason="CPU vectorization not supported")
def test_vector_out_of_bounds():
    # Construct a minimal SPN.
    h1 = Histogram([0., 1., 2., 3.], [0.125, 0.250, 0.625], [1, 1], scope=0)
    h2 = Histogram([0., 1., 2., 3.], [0.100, 0.200, 0.700], [1, 1], scope=1)
    c1 = Categorical(p=[0.1, 0.7, 0.2], scope=0)
    c2 = Categorical(p=[0.4, 0.2, 0.4], scope=1)

    p0 = Product(children=[h1, h2])
    p1 = Product(children=[c1, c2])
    spn = Sum([0.3, 0.7], [p0, p1])

    # Generate some out-of-bounds accesses
    max_index = 2
    inputs = np.column_stack((
        np.random.randint(2 * max_index, size=30),
        np.random.randint(2 * max_index, size=30),
    )).astype("float64")

    if not CPUCompiler.isVectorizationSupported():
        print("Test not supported by the compiler installation")
        return 0

    # Execute the compiled Kernel.
    results = CPUCompiler(verbose=False, computeInLogSpace=False).log_likelihood(spn, inputs, supportMarginal=False)

    # Compute the reference results using the inference from SPFlow.
    reference = log_likelihood(spn, inputs)
    reference = reference.reshape(30)

    # Account for SPFlow behavior: Out-of-bounds values are not returned as -inf, but:
    # for Categoricals: 0.0
    # for Histograms: (np.log(np.finfo(float).eps))
    # Note: np.log(np.finfo(float).eps) is equal to: np.log(2.220446049250313e-16) = -36.04365338911715
    # Find all inputs where an out-of-bounds access will occur.
    # If the access is in-bounds, the corresponding index will be 0 (otherwise: > 0).
    cond = np.sum(np.where((inputs > max_index), 1, 0), axis=1)
    reference[cond > 0] = -np.inf

    # Check the computation results against the reference
    # Check in normal space if log-results are not very close to each other.
    assert np.all(np.isclose(results, reference)) or np.all(np.isclose(np.exp(results), np.exp(reference)))
    

if __name__ == "__main__":
    test_vector_out_of_bounds()
    print("COMPUTATION OK")
