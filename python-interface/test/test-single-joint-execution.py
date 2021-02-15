#!/usr/bin/env python3

import numpy as np
import tempfile
from datetime import datetime
import os
from spn.structure.Base import Product, Sum
from spn.structure.leaves.histogram.Histograms import Histogram
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.structure.Base import get_number_of_nodes, get_number_of_edges
from spn.algorithms.Inference import log_likelihood
from xspn.serialization.binary.BinarySerialization import BinarySerializer, BinaryDeserializer
from xspn.structure.Model import SPNModel
from xspn.structure.Query import JointProbability, ErrorKind
import spncpy as spnc

# Construct a minimal SPN using two Gaussian leaves.
g1 = Gaussian(mean=0.5, stdev=1, scope=0)
g2 = Gaussian(mean=0.125, stdev=0.25, scope=1)
p = Product(children=[g1, g2])

# Wrap the SPN in a model and query.
model = SPNModel(p, "float64", "test")
query = JointProbability(model, batchSize=10)

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
options = dict({"target": "CPU"})
k = compiler.compileQuery(tempfile, options)
# Check that the comiled Kernel actually exists.
if not os.path.isfile(k.fileName()):
    raise RuntimeError("Compilation failed, not Kernel produced")

# Randomly sample input values from the two Gaussian (normal) distributions.
inputs = np.column_stack((np.random.normal(0.5, 1, 30), np.random.normal(0.125, 0.25, 30)))

# Execute the compiled Kernel.
results = k.execute(30, inputs)

# Compute the reference results using the inference from SPFlow.
reference = log_likelihood(p, inputs)
reference = reference.reshape(30)

# Compare computed results and reference to make sure the computation by the compiled Kernel is correct.
if not np.all(np.isclose(results, reference)):
    raise RuntimeError("COMPUTATION FAILED: Results did not match reference!")

print("COMPUTATION OK")

# Remove the serialized SPN file and the compiled Kernel.
os.remove(tempfile)
os.remove(k.fileName())
