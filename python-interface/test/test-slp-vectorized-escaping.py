import numpy as np
import os
import spncpy as spnc
import tempfile
from datetime import datetime
from spn.algorithms.Inference import log_likelihood
from spn.structure.Base import Product, Sum
from spn.structure.leaves.histogram.Histograms import Histogram
from spn.structure.leaves.parametric.Parametric import Categorical
from spn.structure.leaves.parametric.Parametric import Gaussian
from xspn.serialization.binary.BinarySerialization import BinarySerializer
from xspn.structure.Model import SPNModel
from xspn.structure.Query import JointProbability

g0 = Gaussian(mean=0.00, stdev=1, scope=0)
g1 = Gaussian(mean=0.01, stdev=0.75, scope=1)
g2 = Gaussian(mean=0.02, stdev=0.5, scope=2)
g3 = Gaussian(mean=0.03, stdev=0.25, scope=3)
g4 = Gaussian(mean=0.04, stdev=1, scope=4)
g5 = Gaussian(mean=0.05, stdev=0.25, scope=5)
g6 = Gaussian(mean=0.06, stdev=0.5, scope=6)
g7 = Gaussian(mean=0.07, stdev=0.75, scope=7)
g8 = Gaussian(mean=0.08, stdev=1, scope=8)
g9 = Gaussian(mean=0.09, stdev=0.75, scope=9)
g10 = Gaussian(mean=0.10, stdev=1, scope=10)
g11 = Gaussian(mean=0.11, stdev=1, scope=11)

h0 = Histogram([0., 1., 2.], [0.1, 0.9], [1, 1], scope=0)
h1 = Histogram([0., 1., 2.], [0.2, 0.8], [1, 1], scope=1)
h2 = Histogram([0., 1., 2.], [0.3, 0.7], [1, 1], scope=2)
h3 = Histogram([0., 1., 2.], [0.4, 0.6], [1, 1], scope=3)
h4 = Histogram([0., 1., 2.], [0.5, 0.5], [1, 1], scope=4)
h5 = Histogram([0., 1., 2.], [0.6, 0.4], [1, 1], scope=5)
h6 = Histogram([0., 1., 2.], [0.7, 0.3], [1, 1], scope=6)
h7 = Histogram([0., 1., 2.], [0.8, 0.2], [1, 1], scope=7)

c0 = Categorical(p=[0.1, 0.1, 0.8], scope=0)
c1 = Categorical(p=[0.2, 0.2, 0.6], scope=1)
c2 = Categorical(p=[0.3, 0.3, 0.4], scope=2)
c3 = Categorical(p=[0.4, 0.4, 0.2], scope=3)
c4 = Categorical(p=[0.5, 0.4, 0.1], scope=4)
c5 = Categorical(p=[0.6, 0.3, 0.1], scope=5)
c6 = Categorical(p=[0.7, 0.2, 0.1], scope=6)
c7 = Categorical(p=[0.8, 0.1, 0.1], scope=7)

s0 = Sum(children=[g8, h4], weights=[0.5, 0.5])
s1 = Sum(children=[g9, h5], weights=[0.5, 0.5])
s2 = Sum(children=[g10, c6], weights=[0.5, 0.5])
s3 = Sum(children=[g11, h7], weights=[0.5, 0.5])

s4 = Sum(children=[s0, c4], weights=[0.5, 0.5])
s5 = Sum(children=[s1, c5], weights=[0.5, 0.5])
s6 = Sum(children=[s2, g6], weights=[0.5, 0.5])
s7 = Sum(children=[s3, c7], weights=[0.5, 0.5])

s8 = Sum(children=[s4, g4], weights=[0.5, 0.5])
s9 = Sum(children=[s5, g5], weights=[0.5, 0.5])
s10 = Sum(children=[s6, h6], weights=[0.5, 0.5])
s11 = Sum(children=[s7, g7], weights=[0.5, 0.5])

p0 = Product(children=[h0, s8])
p1 = Product(children=[c1, s9])
p2 = Product(children=[c2, s10])
p3 = Product(children=[g3, s11])

p4 = Product(children=[p0, g0])
p5 = Product(children=[p1, g1])
p6 = Product(children=[p2, h2])
p7 = Product(children=[p3, c3])

p8 = Product(children=[p4, c0])
p9 = Product(children=[p5, h1])
p10 = Product(children=[p6, g2])
p11 = Product(children=[p7, h3])

s12 = Sum(children=[p8, p9], weights=[0.5, 0.5])
s13 = Sum(children=[p10, p11], weights=[0.5, 0.5])

s14 = Sum(children=[s12, p2], weights=[0.5, 0.5])
s15 = Sum(children=[s13, s2], weights=[0.5, 0.5])

spn = Product(children=[s14, s15])

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
