# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

import contextlib
import fire
import numpy as np
import os
import shutil
import spnc.spncpy as spncpy
from datetime import datetime
from time import perf_counter_ns
from xspn.serialization.binary.BinarySerialization import BinaryDeserializer, BinarySerializer
from xspn.structure.Query import JointProbability


def translateBool(flag: bool):
    translation = "true" if flag else "false"
    return translation


def measure_execution_time(name: str, spn_file: str, input_data: str, reference_data: str, kernel_dir: str,
                           remove_kernel: bool, vectorize: str, vectorLibrary: str, shuffle: str, maxAttempts=None,
                           maxSuccessfulIterations=None, maxNodeSize=None, maxLookAhead=None,
                           reorderInstructionsDFS=None, allowDuplicateElements=None, allowTopologicalMixing=None):
    # Read the trained SPN from file
    model = BinaryDeserializer(spn_file).deserialize_from_file()
    # Set the name and feature type
    model.name = name
    model.featureType = "float32"
    # Wrap the model in a query.
    query = JointProbability(model=model, batchSize=1, supportMarginal=False)

    # Construct a temporary file that we will use to serialize the SPN to.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmpfile = os.path.join(kernel_dir, f"{name}_{timestamp}.bin")

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

    options = dict({
        "target": "CPU",
        "dump-ir": "false",
        "use-log-space": "true",
        "cpu-vectorize": cpuVectorize,
        "vector-library": vecLib,
        "use-shuffle": useShuffle,
    })

    if maxAttempts is not None:
        options["slp-max-attempts"] = str(maxAttempts)
    if maxSuccessfulIterations is not None:
        options["slp-max-successful-iterations"] = str(maxSuccessfulIterations)
    if maxNodeSize is not None:
        options["slp-max-node-size"] = str(maxNodeSize)
    if maxLookAhead is not None:
        options["slp-max-look-ahead"] = str(maxLookAhead)
    if reorderInstructionsDFS is not None:
        options["slp-reorder-dfs"] = translateBool(bool(reorderInstructionsDFS))
    if allowDuplicateElements is not None:
        options["slp-allow-duplicate-elements"] = translateBool(bool(allowDuplicateElements))
    if allowTopologicalMixing is not None:
        options["slp-allow-topological-mixing"] = translateBool(bool(allowTopologicalMixing))

    compile_start = perf_counter_ns()
    k = compiler.compileQuery(tmpfile, options)
    compile_stop = perf_counter_ns()
    print(f"COMPILATION TIME: {compile_stop - compile_start} ns")
    # Check that the comiled Kernel actually exists.
    if not os.path.isfile(k.fileName()):
        raise RuntimeError("Compilation failed, not Kernel produced")
    if not remove_kernel:
        shutil.copyfile(k.fileName(), os.path.join(kernel_dir, name))

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
    # Suppress FileNotFoundErrors that might occur because the kernels etc. are created in the /tmp/ directory.
    with contextlib.suppress(FileNotFoundError):
        os.remove(tmpfile)
    with contextlib.suppress(FileNotFoundError):
        os.remove(k.fileName())
    if remove_kernel:
        os.remove(os.path.join(kernel_dir, name))


if __name__ == '__main__':
    fire.Fire(measure_execution_time)
