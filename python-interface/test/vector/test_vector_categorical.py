import numpy as np

from spn.structure.Base import Product, Sum
from spn.structure.leaves.parametric.Parametric import Categorical
from spn.algorithms.Inference import log_likelihood

from spnc.cpu import CPUCompiler

def test_vector_categorical():
    # Construct a minimal SPN
    c1 = Categorical(p=[0.35, 0.55, 0.1], scope=0)
    c2 = Categorical(p=[0.25, 0.625, 0.125], scope=1)
    c3 = Categorical(p=[0.5, 0.2, 0.3], scope=2)
    c4 = Categorical(p=[0.6, 0.15, 0.25], scope=3)
    c5 = Categorical(p=[0.7, 0.11, 0.19], scope=4)
    c6 = Categorical(p=[0.8, 0.14, 0.06], scope=5)
    p = Product(children=[c1, c2, c3, c4, c5, c6])

    # Randomly sample input values.
    inputs = np.column_stack((
        np.random.randint(3, size=30),
        np.random.randint(3, size=30),
        np.random.randint(3, size=30),
        np.random.randint(3, size=30),
        np.random.randint(3, size=30),
        np.random.randint(3, size=30),
    )).astype("int32")

    if not CPUCompiler.isVectorizationSupported():
        print("Test not supported by the compiler installation")
        return 0

    # Execute the compiled Kernel.
    results = CPUCompiler(computeInLogSpace=False).log_likelihood(p, inputs, supportMarginal=False)

    # Compute the reference results using the inference from SPFlow.
    reference = log_likelihood(p, inputs)
    reference = reference.reshape(30)

    assert(np.all(np.isclose(results, reference)))
    

if __name__ == "__main__":
    test_vector_categorical()
    print("COMPUTATION OK")
