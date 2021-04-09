import numpy as np

from spn.structure.Base import Product, Sum
from spn.structure.leaves.histogram.Histograms import Histogram
from spn.algorithms.Inference import log_likelihood

from spnc.cpu import CPUCompiler

def test_log_vector_histogram():
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
    results = CPUCompiler().log_likelihood(spn, inputs, supportMarginal=False)

    # Compute the reference results using the inference from SPFlow.
    reference = log_likelihood(spn, inputs)
    reference = reference.reshape(30)

    assert(np.all(np.isclose(results, reference)))
    

if __name__ == "__main__":
    test_log_vector_histogram()
    print("COMPUTATION OK")
