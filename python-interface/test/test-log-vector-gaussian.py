import numpy as np
import tempfile
from datetime import datetime
import os
from spn.structure.Base import Product, Sum
from spn.structure.leaves.parametric.Parametric import Gaussian
from spn.algorithms.Inference import log_likelihood
from xspn.serialization.binary.BinarySerialization import BinarySerializer
from xspn.structure.Model import SPNModel
from xspn.structure.Query import JointProbability
import spncpy as spnc

# Construct a minimal SPN using two Gaussian leaves.
g1 = Gaussian(mean=0.5, stdev=1, scope=0)
g2 = Gaussian(mean=0.125, stdev=0.25, scope=1)
g3 = Gaussian(mean=0.345, stdev=0.24, scope=2)
g4 = Gaussian(mean=0.456, stdev=0.1, scope=3)
g5 = Gaussian(mean=0.94, stdev=0.48, scope=4)
g6 = Gaussian(mean=0.56, stdev=0.42, scope=5)
g7 = Gaussian(mean=0.76, stdev=0.14, scope=6)
g8 = Gaussian(mean=0.32, stdev=0.8, scope=7)
g9 = Gaussian(mean=0.58, stdev=0.9, scope=8)
g10 = Gaussian(mean=0.14, stdev=0.2, scope=9)
p = Product(children=[g1, g2, g3, g4, g5, g6, g7, g8, g9, g10])

# Wrap the SPN in a model and query.
model = SPNModel(p, "float32", "spn_vector")
query = JointProbability(model, batchSize=24)

# Construct a temporary file that we will use to serialize the SPN to.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
tempfile = os.path.join(tempfile.gettempdir(), f"spn_{timestamp}.bin")

# Serialize the SPN to binary format as input to the compiler.
print("Serializing SPN to file...")
BinarySerializer(tempfile).serialize_to_file(query)
# Check that the serialization worked.
if not os.path.isfile(tempfile):
    raise RuntimeError("Serialization of the SPN failed")

print(tempfile)

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
inputs = np.column_stack((np.random.normal(0.5, 1, 30),
                          np.random.normal(0.125, 0.25, 30),
                          np.random.normal(0.345, 0.24, 30),
                          np.random.normal(0.456, 0.1, 30),
                          np.random.normal(0.94, 0.48, 30),
                          np.random.normal(0.56, 0.42, 30),
                          np.random.normal(0.76, 0.14, 30),
                          np.random.normal(0.32, 0.8, 30),
                          np.random.normal(0.58, 0.9, 30),
                          np.random.normal(0.14, 0.2, 30)))

# Execute the compiled Kernel.
results = k.execute(30, inputs.astype("float32"))

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
