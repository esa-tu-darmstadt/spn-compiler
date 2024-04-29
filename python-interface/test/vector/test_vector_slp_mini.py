import numpy as np
import pytest
import time
from spn.algorithms.Inference import log_likelihood
from spn.structure.Base import Sum
from spn.structure.leaves.parametric.Parametric import Gaussian
from spnc.cpu import CPUCompiler, VectorLibrary


@pytest.mark.skipif(
    not CPUCompiler.isVectorizationSupported(), reason="CPU vectorization not supported"
)
def test_vector_slp_mini():
    g0 = Gaussian(mean=0.13, stdev=0.5, scope=0)
    g1 = Gaussian(mean=0.14, stdev=0.25, scope=2)
    g2 = Gaussian(mean=0.11, stdev=1.0, scope=3)
    g3 = Gaussian(mean=0.12, stdev=0.75, scope=1)

    spn = Sum(children=[g0, g1, g2, g3], weights=[0.2, 0.4, 0.1, 0.3])

    # Randomly sample input values from Gaussian (normal) distributions.
    num_samples = 100
    inputs = np.column_stack(
        (
            np.random.normal(loc=0.5, scale=1, size=num_samples),
            np.random.normal(loc=0.125, scale=0.25, size=num_samples),
            np.random.normal(loc=0.345, scale=0.24, size=num_samples),
            np.random.normal(loc=0.456, scale=0.1, size=num_samples),
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
        spnc_dump_ir=True,
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
    test_vector_slp_mini()
    print("COMPUTATION OK")
