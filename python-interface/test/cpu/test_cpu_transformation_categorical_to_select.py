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


def test_cpu_transformation_categorical_to_select():
    # Construct a minimal SPN.
    c1 = Categorical(p=[0.25, 0.75], scope=0)
    c2 = Categorical(p=[0.33, 0.67], scope=1)
    c3 = Categorical(p=[0.80, 0.20], scope=0)
    c4 = Categorical(p=[0.50, 0.50], scope=1)

    p0 = Product(children=[c1, c2])
    p1 = Product(children=[c3, c4])
    spn = Sum([0.3, 0.7], [p0, p1])

    inputs = np.column_stack((
        np.random.randint(2, size=30),
        np.random.randint(2, size=30),
    )).astype("float64")

    # Execute the compiled Kernel.
    results = CPUCompiler(computeInLogSpace=False, vectorize=False).log_likelihood(spn, inputs,
                                                                                   supportMarginal=False,
                                                                                   batchSize=10)

    # Compute the reference results using the inference from SPFlow.
    reference = log_likelihood(spn, inputs)
    reference = reference.reshape(30)

    # Check the computation results against the reference
    # Check in normal space if log-results are not very close to each other.
    assert np.all(np.isclose(results, reference)) or np.all(np.isclose(np.exp(results), np.exp(reference)))
    

if __name__ == "__main__":
    test_cpu_transformation_categorical_to_select()
    print("COMPUTATION OK")
