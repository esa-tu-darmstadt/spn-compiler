import numpy as np
import os
import spncpy as spnc
import tempfile
from datetime import datetime
from spn.algorithms.Inference import log_likelihood
from spn.structure.Base import Sum
from spn.structure.leaves.parametric.Parametric import Gaussian
from xspn.serialization.binary.BinarySerialization import BinarySerializer
from xspn.structure.Model import SPNModel
from xspn.structure.Query import JointProbability

g0 = Gaussian(mean=0.11, stdev=1, scope=0)
g1 = Gaussian(mean=0.12, stdev=0.75, scope=1)
g2 = Gaussian(mean=0.13, stdev=0.5, scope=2)
g3 = Gaussian(mean=0.14, stdev=0.25, scope=3)

s = Sum(children=[g0, g1, g2, g3], weights=[0.25, 0.25, 0.25, 0.25])

# Wrap the SPN in a model and query.
model = SPNModel(s, "float64", "spn_vector")
query = JointProbability(model, batchSize=1)

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

samples = 30

for i in range(samples):

    # Randomly sample input values from Gaussian (normal) distributions.
    inputs = np.column_stack((np.random.normal(0.5, 1),
                              np.random.normal(0.125, 0.25),
                              np.random.normal(0.345, 0.24),
                              np.random.normal(0.456, 0.1)))

    # Execute the compiled Kernel.
    results = k.execute(1, inputs)

    # Compute the reference results using the inference from SPFlow.
    reference = log_likelihood(s, inputs)
    reference = reference.reshape(1)

    print(f"evaluation #{i}: result: {results}, reference: {reference}", end="\r")

    # Compare computed results and reference to make sure the computation by the compiled Kernel is correct.
    if not np.all(np.isclose(results, reference)):
        print("\n")
        raise RuntimeError("COMPUTATION FAILED: Results did not match reference! Input:\n{inputs}")

print("\nCOMPUTATION OK")

# Remove the serialized SPN file and the compiled Kernel.
os.remove(tempfile)
os.remove(k.fileName())
