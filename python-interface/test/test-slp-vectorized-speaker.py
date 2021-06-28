import numpy as np
import os
import tempfile
import time
from datetime import datetime
from spn.algorithms.Inference import log_likelihood
from spnc.cpu import CPUCompiler
from xspn.serialization.binary.BinarySerialization import BinaryDeserializer, BinarySerializer

# Read the trained SPN from file
query = BinaryDeserializer(os.path.join(os.path.dirname(__file__), "..", "..", "mlir", "test", "test-resources",
                                        "speaker_FADG0.bin")).deserialize_from_file()

batchsize = 1

print("feature type:", query.graph.featureType)
print("saved batch size:", query.batchSize)
print("desired batch size:", batchsize)
query.batchSize = batchsize

# Construct a temporary file that we will use to serialize the SPN to.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
tempfile = os.path.join(tempfile.gettempdir(), f"spn_{timestamp}.bin")

# Serialize the SPN to binary format as input to the compiler.
print("Serializing SPN to file...")
BinarySerializer(tempfile).serialize_to_file(query)
# Check that the serialization worked.
if not os.path.isfile(tempfile):
    raise RuntimeError("Serialization of the SPN failed")

num_samples = 2500

# Randomly sample input values from Gaussian (normal) distributions.
inputs = np.column_stack((
    # 26 gaussian inputs
    np.random.normal(0.01, 1.00, num_samples),
    np.random.normal(0.02, 0.90, num_samples),
    np.random.normal(0.03, 0.80, num_samples),
    np.random.normal(0.04, 0.70, num_samples),
    np.random.normal(0.05, 0.60, num_samples),
    np.random.normal(0.06, 0.50, num_samples),
    np.random.normal(0.07, 0.40, num_samples),
    np.random.normal(0.08, 0.30, num_samples),
    np.random.normal(0.09, 0.20, num_samples),
    np.random.normal(0.10, 0.10, num_samples),
    np.random.normal(0.11, 1.00, num_samples),
    np.random.normal(0.12, 0.90, num_samples),
    np.random.normal(0.13, 0.80, num_samples),
    np.random.normal(0.14, 0.70, num_samples),
    np.random.normal(0.15, 0.60, num_samples),
    np.random.normal(0.16, 0.50, num_samples),
    np.random.normal(0.17, 0.40, num_samples),
    np.random.normal(0.18, 0.30, num_samples),
    np.random.normal(0.19, 0.20, num_samples),
    np.random.normal(0.20, 0.10, num_samples),
    np.random.normal(0.21, 1.00, num_samples),
    np.random.normal(0.22, 0.90, num_samples),
    np.random.normal(0.23, 0.80, num_samples),
    np.random.normal(0.24, 0.70, num_samples),
    np.random.normal(0.25, 0.60, num_samples),
    np.random.normal(0.26, 0.50, num_samples),
)).astype("float64")

# Compute the reference results using the inference from SPFlow.
reference = log_likelihood(query.graph.root, inputs)
reference = reference.reshape(num_samples)

# Compile the kernel.
compiler = CPUCompiler(vectorize=True, computeInLogSpace=False)
kernel = compiler.compile_ll(spn=query.graph.root, batchSize=1, supportMarginal=False)

# Execute the compiled Kernel.
time_sum = 0
for i in range(len(reference)):
    # Check the computation results against the reference
    start = time.time()
    result = compiler.execute(kernel, inputs=np.array([inputs[i]]))
    time_sum = time_sum + time.time() - start
    print(f"evaluation #{i}: result: {result[0]:16.8f}, reference: {reference[i]:16.8f}", end='\r')
    assert np.isclose(result, reference[i])
print(f"\nExecution took {time_sum} seconds.")
print("COMPUTATION OK")

# Remove the serialized SPN file and the compiled Kernel.
os.remove(tempfile)
os.remove(kernel.fileName())
