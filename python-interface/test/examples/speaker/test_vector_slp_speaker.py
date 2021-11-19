import numpy as np
import os
import pytest
import time
from spn.algorithms.Inference import log_likelihood
from spnc.cpu import CPUCompiler
from xspn.serialization.binary.BinarySerialization import BinaryDeserializer


@pytest.mark.skipif(not CPUCompiler.isVectorizationSupported(), reason="CPU vectorization not supported")
def test_vector_slp_speaker():
    # Locate test resources located in same directory as this script.
    scriptPath = os.path.realpath(os.path.dirname(__file__))

    # Read the trained SPN from file
    model = BinaryDeserializer(os.path.join(scriptPath, "speaker_FADG0.bin")).deserialize_from_file()
    spn = model.graph.root

    # Randomly sample input values from Gaussian (normal) distributions.
    num_samples = 10000
    inputs = np.column_stack((
        # 26 gaussian inputs
        np.random.normal(loc=0.01, scale=1.00, size=num_samples),
        np.random.normal(loc=0.02, scale=0.90, size=num_samples),
        np.random.normal(loc=0.03, scale=0.80, size=num_samples),
        np.random.normal(loc=0.04, scale=0.70, size=num_samples),
        np.random.normal(loc=0.05, scale=0.60, size=num_samples),
        np.random.normal(loc=0.06, scale=0.50, size=num_samples),
        np.random.normal(loc=0.07, scale=0.40, size=num_samples),
        np.random.normal(loc=0.08, scale=0.30, size=num_samples),
        np.random.normal(loc=0.09, scale=0.20, size=num_samples),
        np.random.normal(loc=0.10, scale=0.10, size=num_samples),
        np.random.normal(loc=0.11, scale=1.00, size=num_samples),
        np.random.normal(loc=0.12, scale=0.90, size=num_samples),
        np.random.normal(loc=0.13, scale=0.80, size=num_samples),
        np.random.normal(loc=0.14, scale=0.70, size=num_samples),
        np.random.normal(loc=0.15, scale=0.60, size=num_samples),
        np.random.normal(loc=0.16, scale=0.50, size=num_samples),
        np.random.normal(loc=0.17, scale=0.40, size=num_samples),
        np.random.normal(loc=0.18, scale=0.30, size=num_samples),
        np.random.normal(loc=0.19, scale=0.20, size=num_samples),
        np.random.normal(loc=0.20, scale=0.10, size=num_samples),
        np.random.normal(loc=0.21, scale=1.00, size=num_samples),
        np.random.normal(loc=0.22, scale=0.90, size=num_samples),
        np.random.normal(loc=0.23, scale=0.80, size=num_samples),
        np.random.normal(loc=0.24, scale=0.70, size=num_samples),
        np.random.normal(loc=0.25, scale=0.60, size=num_samples),
        np.random.normal(loc=0.26, scale=0.50, size=num_samples),
    )).astype("float64")

    # Compute the reference results using the inference from SPFlow.
    reference = log_likelihood(spn, inputs)
    reference = reference.reshape(num_samples)

    # Compile the kernel with batch size 1 to enable SLP vectorization.
    compiler = CPUCompiler(vectorize=True, computeInLogSpace=True)
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
    test_vector_slp_speaker()
    print("COMPUTATION OK")
