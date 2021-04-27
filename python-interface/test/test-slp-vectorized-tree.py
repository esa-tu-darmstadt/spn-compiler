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

g0 = Gaussian(mean=0.11, stdev=1, scope=0)
g1 = Gaussian(mean=0.12, stdev=0.75, scope=1)
g2 = Gaussian(mean=0.13, stdev=0.5, scope=2)
g3 = Gaussian(mean=0.14, stdev=0.25, scope=3)
g4 = Gaussian(mean=0.15, stdev=1, scope=4)
g5 = Gaussian(mean=0.16, stdev=0.25, scope=5)
g6 = Gaussian(mean=0.17, stdev=0.5, scope=6)
g7 = Gaussian(mean=0.18, stdev=0.75, scope=7)
g8 = Gaussian(mean=0.19, stdev=1, scope=8)

p0 = Product(children=[g0, g1, g2, g4])
p1 = Product(children=[g3, g4, g4, g5])
p2 = Product(children=[g6, g4, g7, g8])
p3 = Product(children=[g8, g6, g4, g2])

s0 = Sum(children=[g0, g1, g2, p0], weights=[0.25, 0.25, 0.25, 0.25])
s1 = Sum(children=[g3, g4, g5, p1], weights=[0.25, 0.25, 0.25, 0.25])
s2 = Sum(children=[g6, g7, g8, p2], weights=[0.25, 0.25, 0.25, 0.25])
s3 = Sum(children=[g0, g4, g8, p3], weights=[0.25, 0.25, 0.25, 0.25])

spn = Product(children=[s0, s1, s2, s3])

# Wrap the SPN in a model and query.
model = SPNModel(spn, "float64", "spn_vector")
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
		                  np.random.normal(0.456, 0.1),
		                  np.random.normal(0.94, 0.48),
		                  np.random.normal(0.56, 0.42),
		                  np.random.normal(0.76, 0.14),
		                  np.random.normal(0.32, 0.58),
		                  np.random.normal(0.58, 0.219),
		                  np.random.normal(0.14, 0.52),
		                  np.random.normal(0.24, 0.42),
		                  np.random.normal(0.34, 0.1),
		                  np.random.normal(0.44, 0.9),
		                  np.random.normal(0.54, 0.7),
		                  np.random.normal(0.64, 0.5),
		                  np.random.normal(0.74, 0.4)))

	# Execute the compiled Kernel.
	results = k.execute(1, inputs)

	# Compute the reference results using the inference from SPFlow.
	reference = log_likelihood(spn, inputs)
	reference = reference.reshape(1)

	print(f"evaluation #{i}: result: {results}, reference: {reference}")

	# Compare computed results and reference to make sure the computation by the compiled Kernel is correct.
	if not np.all(np.isclose(results, reference)):
	    raise RuntimeError("COMPUTATION FAILED: Results did not match reference!")

print("COMPUTATION OK")

# Remove the serialized SPN file and the compiled Kernel.
os.remove(tempfile)
os.remove(k.fileName())
