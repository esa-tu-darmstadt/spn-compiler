import numpy as np
import tempfile
from datetime import datetime
import os
from spn.structure.Base import Product, Sum
from spn.structure.leaves.histogram.Histograms import Histogram
from spn.algorithms.Inference import log_likelihood
from xspn.serialization.binary.BinarySerialization import BinarySerializer, BinaryDeserializer
from xspn.structure.Model import SPNModel
from xspn.structure.Query import JointProbability, ErrorKind
import spncpy as spnc

# Construct a minimal SPN using two Gaussian leaves.
h1 = Histogram([0., 1., 2.], [0.25, 0.75], [1, 1], scope=0)
h2 = Histogram([0., 1., 2.], [0.45, 0.55], [1, 1], scope=1)
h3 = Histogram([0., 1., 2.], [0.33, 0.67], [1, 1], scope=0)
h4 = Histogram([0., 1., 2.], [0.875, 0.125], [1, 1], scope=1)

p0 = Product(children=[h1, h2])
p1 = Product(children=[h3, h4])
spn = Sum([0.3, 0.7], [p0, p1])

# Wrap the SPN in a model and query.
model = SPNModel(spn, "float64", "spn_vector")
query = JointProbability(model, batchSize=16)

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

# Randomly sample input values from the two Gaussian (normal) distributions.
inputs = np.column_stack((
    np.random.randint(2, size=30),
    np.random.randint(2, size=30),
)).astype("float64")

# Execute the compiled Kernel.
results = k.execute(30, inputs)

# Compute the reference results using the inference from SPFlow.
reference = log_likelihood(spn, inputs)
reference = reference.reshape(30)

print(results)

print(reference)

# Compare computed results and reference to make sure the computation by the compiled Kernel is correct.
if not np.all(np.isclose(results, reference)):
    raise RuntimeError("COMPUTATION FAILED: Results did not match reference!")

print("COMPUTATION OK")

# Remove the serialized SPN file and the compiled Kernel.
os.remove(tempfile)
os.remove(k.fileName())
