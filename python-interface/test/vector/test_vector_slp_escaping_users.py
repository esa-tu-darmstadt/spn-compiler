import numpy as np
import pytest
import time
from spn.algorithms.Inference import log_likelihood
from spn.structure.Base import Product, Sum
from spn.structure.leaves.histogram.Histograms import Histogram
from spn.structure.leaves.parametric.Parametric import Categorical
from spn.structure.leaves.parametric.Parametric import Gaussian
from spnc.cpu import CPUCompiler


@pytest.mark.skipif(not CPUCompiler.isVectorizationSupported(), reason="CPU vectorization not supported")
def test_vector_slp_escaping_users():
    g0 = Gaussian(mean=0.00, stdev=1, scope=0)
    g1 = Gaussian(mean=0.01, stdev=0.75, scope=1)
    g2 = Gaussian(mean=0.02, stdev=0.5, scope=2)
    g3 = Gaussian(mean=0.03, stdev=0.25, scope=3)
    g4 = Gaussian(mean=0.04, stdev=1, scope=4)
    g5 = Gaussian(mean=0.05, stdev=0.25, scope=5)
    g6 = Gaussian(mean=0.06, stdev=0.5, scope=6)
    g7 = Gaussian(mean=0.07, stdev=0.75, scope=7)
    g8 = Gaussian(mean=0.08, stdev=1, scope=8)
    g9 = Gaussian(mean=0.09, stdev=0.75, scope=9)
    g10 = Gaussian(mean=0.10, stdev=1, scope=10)
    g11 = Gaussian(mean=0.11, stdev=1, scope=11)

    h0 = Histogram([0., 1., 2.], [0.1, 0.9], [1, 1], scope=12)
    h1 = Histogram([0., 1., 2.], [0.2, 0.8], [1, 1], scope=13)
    h2 = Histogram([0., 1., 2.], [0.3, 0.7], [1, 1], scope=14)
    h3 = Histogram([0., 1., 2.], [0.4, 0.6], [1, 1], scope=15)
    h4 = Histogram([0., 1., 2.], [0.5, 0.5], [1, 1], scope=16)
    h5 = Histogram([0., 1., 2.], [0.6, 0.4], [1, 1], scope=17)
    h6 = Histogram([0., 1., 2.], [0.7, 0.3], [1, 1], scope=18)
    h7 = Histogram([0., 1., 2.], [0.8, 0.2], [1, 1], scope=19)

    c0 = Categorical(p=[0.1, 0.1, 0.8], scope=20)
    c1 = Categorical(p=[0.2, 0.2, 0.6], scope=21)
    c2 = Categorical(p=[0.3, 0.3, 0.4], scope=22)
    c3 = Categorical(p=[0.4, 0.4, 0.2], scope=23)
    c4 = Categorical(p=[0.5, 0.4, 0.1], scope=24)
    c5 = Categorical(p=[0.6, 0.3, 0.1], scope=25)
    c6 = Categorical(p=[0.7, 0.2, 0.1], scope=26)
    c7 = Categorical(p=[0.8, 0.1, 0.1], scope=27)

    s0 = Sum(children=[g8, h4], weights=[0.5, 0.5])
    s1 = Sum(children=[g9, h5], weights=[0.5, 0.5])
    s2 = Sum(children=[g10, c6], weights=[0.5, 0.5])
    s3 = Sum(children=[g11, h7], weights=[0.5, 0.5])

    s4 = Sum(children=[s0, c4], weights=[0.5, 0.5])
    s5 = Sum(children=[s1, c5], weights=[0.5, 0.5])
    s6 = Sum(children=[s2, g6], weights=[0.5, 0.5])
    s7 = Sum(children=[s3, c7], weights=[0.5, 0.5])

    s8 = Sum(children=[s4, g4], weights=[0.5, 0.5])
    s9 = Sum(children=[s5, g5], weights=[0.5, 0.5])
    s10 = Sum(children=[s6, h6], weights=[0.5, 0.5])
    s11 = Sum(children=[s7, g7], weights=[0.5, 0.5])

    p0 = Product(children=[h0, s8])
    p1 = Product(children=[c1, s9])
    p2 = Product(children=[c2, s10])
    p3 = Product(children=[g3, s11])

    p4 = Product(children=[p0, g0])
    p5 = Product(children=[p1, g1])
    p6 = Product(children=[p2, h2])
    p7 = Product(children=[p3, c3])

    p8 = Product(children=[p4, c0])
    p9 = Product(children=[p5, h1])
    p10 = Product(children=[p6, g2])
    p11 = Product(children=[p7, h3])

    s12 = Sum(children=[p8, p9], weights=[0.5, 0.5])
    s13 = Sum(children=[p10, p11], weights=[0.5, 0.5])

    s14 = Sum(children=[s12, p2], weights=[0.5, 0.5])
    s15 = Sum(children=[s13, s2], weights=[0.5, 0.5])

    spn = Product(children=[s14, s15])

    # Randomly sample input values from Gaussian (normal) distributions.
    num_samples = 100
    inputs = np.column_stack((
        # gaussian
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
        # histogram
        np.random.randint(low=0, high=2, size=num_samples),
        np.random.randint(low=0, high=2, size=num_samples),
        np.random.randint(low=0, high=2, size=num_samples),
        np.random.randint(low=0, high=2, size=num_samples),
        np.random.randint(low=0, high=2, size=num_samples),
        np.random.randint(low=0, high=2, size=num_samples),
        np.random.randint(low=0, high=2, size=num_samples),
        np.random.randint(low=0, high=2, size=num_samples),
        # categorical
        np.random.randint(low=0, high=3, size=num_samples),
        np.random.randint(low=0, high=3, size=num_samples),
        np.random.randint(low=0, high=3, size=num_samples),
        np.random.randint(low=0, high=3, size=num_samples),
        np.random.randint(low=0, high=3, size=num_samples),
        np.random.randint(low=0, high=3, size=num_samples),
        np.random.randint(low=0, high=3, size=num_samples),
        np.random.randint(low=0, high=3, size=num_samples))).astype("float64")

    # Compute the reference results using the inference from SPFlow.
    reference = log_likelihood(spn, inputs)
    reference = reference.reshape(num_samples)

    # Compile the kernel with batch size 1 to enable SLP vectorization.
    compiler = CPUCompiler(vectorize=True, computeInLogSpace=True, vectorLibrary="LIBMVEC")
    kernel = compiler.compile_ll(spn=spn, batchSize=1, supportMarginal=False)

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
            raise AssertionError()
    print(f"\nExecution of {len(reference)} samples took {time_sum} seconds.")


if __name__ == "__main__":
    test_vector_slp_escaping_users()
    print("COMPUTATION OK")
