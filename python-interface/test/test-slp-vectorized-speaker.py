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
reference = log_likelihood(query.graph.root, inputs)
reference = reference.reshape(num_samples)

# Compile the kernel.
compiler = CPUCompiler(vectorize=True, computeInLogSpace=True)
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
print(f"\nExecution of {len(reference)} samples took {time_sum} seconds.")
print("COMPUTATION OK")

# Remove the serialized SPN file and the compiled Kernel.
os.remove(tempfile)
os.remove(kernel.fileName())
