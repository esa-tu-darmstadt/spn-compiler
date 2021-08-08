# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

import fire
import os
import spnc.spncpy as spncpy
import tempfile
from datetime import datetime
from xspn.serialization.binary.BinarySerialization import BinaryDeserializer, BinarySerializer
from xspn.structure.Query import JointProbability


def translateBool(flag: bool):
    translation = "true" if flag else "false"
    return translation


def compute_sizes(name: str, spn_file: str, vectorize: str, vectorLibrary: str, shuffle: str):
    # Read the trained SPN from file
    model = BinaryDeserializer(spn_file).deserialize_from_file()

    # Set the name and feature type
    model.name = name
    model.featureType = "float32"
    # Wrap the model in a query.
    query = JointProbability(model=model, batchSize=1, supportMarginal=False)

    # Construct a temporary file that we will use to serialize the SPN to.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmpfile = os.path.join(tempfile.gettempdir(), f"{name}_{timestamp}.bin")

    # Serialize the SPN to binary format as input to the compiler.
    BinarySerializer(tmpfile).serialize_to_file(query)
    # Check that the serialization worked.
    if not os.path.isfile(tmpfile):
        raise RuntimeError("Serialization of the SPN failed")

    # Invoke the compiler.
    compiler = spncpy.SPNCompiler()

    # Compile the query into a Kernel.
    cpuVectorize = translateBool(bool(vectorize))
    print(f"CPU Vectorize {cpuVectorize}")
    vecLib = "LIBMVEC" if bool(vectorLibrary) else "None"
    useShuffle = translateBool(bool(shuffle))
    options = dict({"target": "CPU", "dump-ir": "false", "use-log-space": "true", "cpu-vectorize": cpuVectorize,
                    "vector-library": vecLib, "use-shuffle": useShuffle})

    k = compiler.compileQuery(tmpfile, options)
    # Check that the compiled kernel actually exists.
    if not os.path.isfile(k.fileName()):
        raise RuntimeError("Compilation failed, no kernel produced")

    print("STATUS OK")

    # Remove the serialized SPN file and the compiled Kernel.
    os.remove(tmpfile)
    os.remove(k.fileName())


if __name__ == '__main__':
    fire.Fire(compute_sizes)
