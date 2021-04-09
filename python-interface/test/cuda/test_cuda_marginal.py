import numpy as np

from spn.algorithms.Inference import log_likelihood
from spn.structure.Base import Product
from spn.structure.leaves.parametric.Parametric import Gaussian

from spnc.gpu import CUDACompiler

def test_cuda_marginal_gaussian():

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

    # Randomly sample input values from the two Gaussian (normal) distributions.
    inputs = np.column_stack((np.random.normal(0.5, 1, 30),
                            np.random.normal(0.125, 0.25, 30),
                            np.random.normal(0.345, 0.24, 30),
                            np.random.normal(0.456, 0.1, 30),
                            np.random.normal(0.94, 0.48, 30),
                            np.random.normal(0.56, 0.42, 30),
                            np.random.normal(0.76, 0.14, 30),
                            np.random.normal(0.32, 0.8, 30),
                            np.random.normal(0.58, 0.9, 30),
                            np.random.normal(0.14, 0.2, 30))).astype("float32")

    inputs.ravel()[np.random.choice(inputs.size, 15, replace=False)] = np.nan

    if not CUDACompiler.isAvailable():
        print("Test not supported by the compiler installation")
        return 0

    # Execute the compiled Kernel.
    results = CUDACompiler().log_likelihood(p, inputs)

    # Compute the reference results using the inference from SPFlow.
    reference = log_likelihood(p, inputs)
    reference = reference.reshape(30)

    # Compare computed results and reference to make sure the computation by the compiled Kernel is correct.
    assert(np.all(np.isclose(results, reference)))

if __name__ == "__main__":
    test_cuda_marginal_gaussian()
    print("COMPUTATION OK")
