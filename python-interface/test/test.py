import os

import numpy as np
import spncpy as spnc

k = spnc.Kernel(os.environ["DYNAMIC_BINARY_HOME"]+"/libdynamic-load-test.so", "foo")
print(k.fileName())
sample = np.array([1, 2, 3, 4, 5]).astype(int)
result = k.execute(5, sample)

print(result)