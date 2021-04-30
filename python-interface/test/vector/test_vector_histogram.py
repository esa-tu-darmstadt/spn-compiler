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
from spn.algorithms.Inference import log_likelihood

from spnc.cpu import CPUCompiler


def test_vector_histogram():
    # Construct a minimal SPN.
    h1 = Histogram([0., 1., 2.], [0.25, 0.75], [1, 1], scope=0)
    h2 = Histogram([0., 1., 2.], [0.45, 0.55], [1, 1], scope=1)
    h3 = Histogram([0., 1., 2.], [0.33, 0.67], [1, 1], scope=0)
    h4 = Histogram([0., 1., 2.], [0.875, 0.125], [1, 1], scope=1)

    p0 = Product(children=[h1, h2])
    p1 = Product(children=[h3, h4])
    spn = Sum([0.3, 0.7], [p0, p1])

    inputs = np.column_stack((
        np.random.randint(2, size=30),
        np.random.randint(2, size=30),
    )).astype("float64")

    if not CPUCompiler.isVectorizationSupported():
        print("Test not supported by the compiler installation")
        return 0

    # Execute the compiled Kernel.
    results = CPUCompiler(computeInLogSpace=False).log_likelihood(spn, inputs, supportMarginal=False)

    # Compute the reference results using the inference from SPFlow.
    reference = log_likelihood(spn, inputs)
    reference = reference.reshape(30)

    # Check the computation results against the reference
    # Check in normal space if log-results are not very close to each other.
    assert np.all(np.isclose(results, reference)) or np.all(np.isclose(np.exp(results), np.exp(reference)))
    

if __name__ == "__main__":
    test_vector_histogram()
    print("COMPUTATION OK")
