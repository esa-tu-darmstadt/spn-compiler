#!/usr/bin/env python3

import fire
import os
import spncpy as spnc
import subprocess
import tempfile
from datetime import datetime
from xspn.serialization.binary.BinarySerialization import BinaryDeserializer, BinarySerializer
from xspn.structure.Query import JointProbability


def translateBool(flag: bool):
    translation = "true" if flag else "false"
    return translation


def estimate_costs(name: str, spn_file: str, vectorize: str, vectorLibrary: str, shuffle: str, mca_iterations: str):
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
    compiler = spnc.SPNCompiler()

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

    kernel_file = k.fileName()

    command = ["llvm-objdump", "--disassemble-symbols=vec_task_0", "--no-leading-addr", "--no-show-raw-insn",
               "--no-print-imm-hex", kernel_file]
    objdump = subprocess.run(command, capture_output=True, text=True)
    if objdump.returncode != 0:
        print("llvm-objdump failed")
        print(objdump.stdout)
        print(objdump.stderr)
        return
    # Replace 'callq 0x1234 <logf@plt>' etc. with 'call 0x1234'
    command = ["sed", "-E", "s/callq\\s+(0x[0-9]+)\\s+<([a-zA-Z0-9]+)@plt>/call  \\1/"]
    print(command)
    objdump = subprocess.run(command, capture_output=True, input=objdump.stdout, text=True)
    if objdump.returncode != 0:
        print("sed failed")
        print(objdump.stdout)
        print(objdump.stderr)
        return
    command = ["llvm-mca", "--iterations", str(mca_iterations)]
    mca_output = subprocess.run(command, capture_output=True, input=objdump.stdout, text=True)
    if mca_output.returncode != 0:
        print("llvm-mca failed")
        print(mca_output.stdout)
        print(mca_output.stderr)
        return
    print(mca_output.stdout)
    print("STATUS OK")

    # Remove the serialized SPN file and the compiled Kernel.
    os.remove(tmpfile)
    os.remove(k.fileName())


if __name__ == '__main__':
    fire.Fire(estimate_costs)
