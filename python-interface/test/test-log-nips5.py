import os
import tempfile
from datetime import datetime

import numpy as np
from xspn.serialization.binary.BinarySerialization import BinaryDeserializer, BinarySerializer
import spncpy as spnc

# Read the trained SPN from file
single = BinaryDeserializer("../../mlir/test/test-resources/NIPS5.bin").deserialize_from_file()

print(single.batchSize)
print(single.graph.featureType)
query = single
query.batchSize = 36
print(query.batchSize)

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
options = dict({"target": "CPU", "cpu-vectorize": "true", "use-log-space": "true"})
k = compiler.compileQuery(tempfile, options)
# Check that the comiled Kernel actually exists.
if not os.path.isfile(k.fileName()):
    raise RuntimeError("Compilation failed, not Kernel produced")

# Read input data from inputdata.txt
inputs = np.genfromtxt("inputdata.txt", delimiter=";", dtype="int32")
print(inputs.shape)

# Execute the compiled Kernel.
results = k.execute(10000, inputs)

reference = np.genfromtxt("outputdata.txt", delimiter=";", dtype="float64")
print(reference.shape)
reference.reshape(10000)

# Compare computed results and reference to make sure the computation by the compiled Kernel is correct.
if not np.all(np.isclose(results, reference[:10000])):
    raise RuntimeError("COMPUTATION FAILED: Results did not match reference!")

print("COMPUTATION OK")

# Remove the serialized SPN file and the compiled Kernel.
os.remove(tempfile)
os.remove(k.fileName())
