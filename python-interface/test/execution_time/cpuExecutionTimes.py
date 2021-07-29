#!/usr/bin/env python3

import fire
import numpy as np
import os
import tempfile
import spnc.spncpy as spncpy
from datetime import datetime
from time import perf_counter_ns
from xspn.serialization.binary.BinarySerialization import BinaryDeserializer, BinarySerializer
from xspn.structure.Query import JointProbability


def translateBool(flag: bool):
    translation = "true" if flag else "false"
    return translation


def measure_execution_time(name: str, spn_file: str, input_data: str, reference_data: str, vectorize: str,
                           vectorLibrary: str, shuffle: str):
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

    compile_start = perf_counter_ns()
    k = compiler.compileQuery(tmpfile, options)
    compile_stop = perf_counter_ns()
    print(f"COMPILATION TIME: {compile_stop - compile_start} ns")
    # Check that the comiled Kernel actually exists.
    if not os.path.isfile(k.fileName()):
        raise RuntimeError("Compilation failed, not Kernel produced")

    # Read input and reference data
    inputs = np.genfromtxt(input_data, delimiter=",", dtype="float32")
    reference = np.genfromtxt(reference_data, delimiter=",", dtype="float32")

    # Execute the compiled Kernel.
    numSamples = inputs.shape[0]
    print("Comparing with reference...")
    for i in range(numSamples):
        results = k.execute(1, np.atleast_2d(inputs[i]))
        # Compare computed result and reference to make sure the computation by the compiled Kernel is correct.
        if not np.all(np.isclose(results, reference[i])):
            raise RuntimeError("COMPUTATION FAILED: Results did not match reference!")

    print("STATUS OK")

    # Remove the serialized SPN file and the compiled Kernel.
    os.remove(tmpfile)
    os.remove(k.fileName())


if __name__ == '__main__':
    fire.Fire(measure_execution_time)
