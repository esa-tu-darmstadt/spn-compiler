import numpy as np
import os
import spncpy as spnc
import tempfile
import time
from datetime import datetime
from spn.algorithms.Inference import log_likelihood
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

# Invoke the compiler.
print("Invoking compiler...")
compiler = spnc.SPNCompiler()
# Compile the query into a Kernel.
options = dict({"target": "CPU", "cpu-vectorize": "true"})
k = compiler.compileQuery(tempfile, options)
# Check that the comiled Kernel actually exists.
if not os.path.isfile(k.fileName()):
    raise RuntimeError("Compilation failed, not Kernel produced")

samples = 2500

time_sum = 0

for i in range(samples):

    # Randomly sample input values from Gaussian (normal) distributions.
    inputs = np.column_stack((
        # 26 gaussian inputs
        np.random.normal(0.01, 1.00),
        np.random.normal(0.02, 0.90),
        np.random.normal(0.03, 0.80),
        np.random.normal(0.04, 0.70),
        np.random.normal(0.05, 0.60),
        np.random.normal(0.06, 0.50),
        np.random.normal(0.07, 0.40),
        np.random.normal(0.08, 0.30),
        np.random.normal(0.09, 0.20),
        np.random.normal(0.10, 0.10),
        np.random.normal(0.11, 1.00),
        np.random.normal(0.12, 0.90),
        np.random.normal(0.13, 0.80),
        np.random.normal(0.14, 0.70),
        np.random.normal(0.15, 0.60),
        np.random.normal(0.16, 0.50),
        np.random.normal(0.17, 0.40),
        np.random.normal(0.18, 0.30),
        np.random.normal(0.19, 0.20),
        np.random.normal(0.20, 0.10),
        np.random.normal(0.21, 1.00),
        np.random.normal(0.22, 0.90),
        np.random.normal(0.23, 0.80),
        np.random.normal(0.24, 0.70),
        np.random.normal(0.25, 0.60),
        np.random.normal(0.26, 0.50),
    )).astype("float64")

    # Execute the compiled Kernel.
    start = time.time()
    result = k.execute(1, inputs)
    time_sum = time_sum + time.time() - start
    # Compute the reference results using the inference from SPFlow.
    reference = log_likelihood(query.graph.root, inputs)
    reference = reference.reshape(1)

    print(f"evaluation #{i}: result: {result[0]:16.8f}, reference: {reference[0]:16.8f}", end='\r')

    # Compare computed results and reference to make sure the computation by the compiled Kernel is correct.
    if not np.all(np.isclose(result, reference)):
        print("\n")
        raise RuntimeError(f"COMPUTATION FAILED: Results did not match reference! Input:\n{inputs}")

print(f"\nExecution took {time_sum} seconds.")
print("\nCOMPUTATION OK")

# Remove the serialized SPN file and the compiled Kernel.
os.remove(tempfile)
# os.remove(k.fileName())
