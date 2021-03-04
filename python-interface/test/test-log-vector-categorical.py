import numpy as np
import tempfile
from datetime import datetime
import os
from spn.structure.Base import Product
from spn.structure.leaves.parametric.Parametric import Categorical
from spn.algorithms.Inference import log_likelihood
from xspn.serialization.binary.BinarySerialization import BinarySerializer
from xspn.structure.Model import SPNModel
from xspn.structure.Query import JointProbability
import spncpy as spnc

# Construct a minimal SPN using two Gaussian leaves.
c1 = Categorical(p=[0.35, 0.55, 0.1], scope=0)
c2 = Categorical(p=[0.25, 0.625, 0.125], scope=1)
c3 = Categorical(p=[0.5, 0.2, 0.3], scope=2)
c4 = Categorical(p=[0.6, 0.15, 0.25], scope=3)
c5 = Categorical(p=[0.7, 0.11, 0.19], scope=4)
c6 = Categorical(p=[0.8, 0.14, 0.06], scope=5)
p = Product(children=[c1, c2, c3, c4, c5, c6])

# Wrap the SPN in a model and query.
model = SPNModel(p, "int32", "spn_vector")
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
    np.random.randint(3, size=30),
    np.random.randint(3, size=30),
    np.random.randint(3, size=30),
    np.random.randint(3, size=30),
    np.random.randint(3, size=30),
    np.random.randint(3, size=30),
)).astype("int32")

# Execute the compiled Kernel.
results = k.execute(30, inputs)

# Compute the reference results using the inference from SPFlow.
reference = log_likelihood(p, inputs)
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
