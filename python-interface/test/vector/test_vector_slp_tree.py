import numpy as np
import pytest
import time
from spn.algorithms.Inference import log_likelihood
from spn.structure.Base import Product, Sum
from spn.structure.leaves.parametric.Parametric import Gaussian
from spnc.cpu import CPUCompiler, VectorLibrary


@pytest.mark.skipif(
    not CPUCompiler.isVectorizationSupported(), reason="CPU vectorization not supported"
)
def test_vector_slp_tree():
    g0 = Gaussian(mean=0.11, stdev=1, scope=0)
    g1 = Gaussian(mean=0.12, stdev=0.75, scope=1)
    g2 = Gaussian(mean=0.13, stdev=0.5, scope=2)
    g3 = Gaussian(mean=0.14, stdev=0.25, scope=3)
    g4 = Gaussian(mean=0.15, stdev=1, scope=4)
    g5 = Gaussian(mean=0.16, stdev=0.25, scope=5)
    g6 = Gaussian(mean=0.17, stdev=0.5, scope=6)
    g7 = Gaussian(mean=0.18, stdev=0.75, scope=7)
    g8 = Gaussian(mean=0.19, stdev=1, scope=8)

    p0 = Product(children=[g0, g1, g2, g4])
    p1 = Product(children=[g3, g4, g4, g5])
    p2 = Product(children=[g6, g4, g7, g8])
    p3 = Product(children=[g8, g6, g4, g2])

    s0 = Sum(children=[g0, g1, g2, p0], weights=[0.25, 0.25, 0.25, 0.25])
    s1 = Sum(children=[g3, g4, g5, p1], weights=[0.25, 0.25, 0.25, 0.25])
    s2 = Sum(children=[g6, g7, g8, p2], weights=[0.25, 0.25, 0.25, 0.25])
    s3 = Sum(children=[g0, g4, g8, p3], weights=[0.25, 0.25, 0.25, 0.25])

    spn = Product(children=[s0, s1, s2, s3])

    # Randomly sample input values from Gaussian (normal) distributions.
    num_samples = 100
    inputs = np.column_stack(
        (
            np.random.normal(loc=0.5, scale=1, size=num_samples),
            np.random.normal(loc=0.125, scale=0.25, size=num_samples),
            np.random.normal(loc=0.345, scale=0.24, size=num_samples),
            np.random.normal(loc=0.456, scale=0.1, size=num_samples),
            np.random.normal(loc=0.94, scale=0.48, size=num_samples),
            np.random.normal(loc=0.56, scale=0.42, size=num_samples),
            np.random.normal(loc=0.76, scale=0.14, size=num_samples),
            np.random.normal(loc=0.32, scale=0.58, size=num_samples),
            np.random.normal(loc=0.58, scale=0.219, size=num_samples),
            np.random.normal(loc=0.14, scale=0.52, size=num_samples),
            np.random.normal(loc=0.24, scale=0.42, size=num_samples),
            np.random.normal(loc=0.34, scale=0.1, size=num_samples),
            np.random.normal(loc=0.44, scale=0.9, size=num_samples),
            np.random.normal(loc=0.54, scale=0.7, size=num_samples),
            np.random.normal(loc=0.64, scale=0.5, size=num_samples),
            np.random.normal(loc=0.74, scale=0.4, size=num_samples),
        )
    ).astype("float64")

    # Compute the reference results using the inference from SPFlow.
    reference = log_likelihood(spn, inputs)
    reference = reference.reshape(num_samples)

    # Compile the kernel with batch size 1 to enable SLP vectorization.
    compiler = CPUCompiler(
        spnc_cpu_vectorize=True,
        spnc_use_log_space=True,
        spnc_vector_library=VectorLibrary.LIBMVEC,
    )
    kernel = compiler.compile_ll(spn=spn, batchSize=1, supportMarginal=False)

    # Execute the compiled Kernel.
    time_sum = 0
    for i in range(len(reference)):
        # Check the computation results against the reference
        start = time.time()
        result = compiler.execute(kernel, inputs=np.array([inputs[i]]))
        time_sum = time_sum + time.time() - start
        print(
            f"evaluation #{i}: result: {result[0]:16.8f}, reference: {reference[i]:16.8f}",
            end="\r",
        )
        if not np.isclose(result, reference[i]):
            print(
                f"\nevaluation #{i} failed: result: {result[0]:16.8f}, reference: {reference[i]:16.8f}"
            )
            raise AssertionError()
    print(f"\nExecution of {len(reference)} samples took {time_sum} seconds.")


if __name__ == "__main__":
    test_vector_slp_tree()
    print("COMPUTATION OK")
