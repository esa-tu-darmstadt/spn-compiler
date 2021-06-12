# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

import csv
import numpy as np
import os
import time
from pathlib import Path
from spnc.cpu import CPUCompiler
from xspn.serialization.binary.BinarySerialization import BinaryDeserializer


def test_spn(spn, inputs, reference, spn_name, log_file):
    # Things to log.
    log_dict = {
        "name": spn_name,
        "execution time (s)": 0,
        "compile time (s)": 0,
        "num_evaluations": reference.size
    }
    # Compile the kernel.
    compiler = CPUCompiler(vectorize=False, computeInLogSpace=True)
    start = time.time()
    kernel = compiler.compile_ll(spn=spn, batchSize=1, supportMarginal=False)
    log_dict["compile time (s)"] = time.time() - start
    # Execute the compiled kernel.
    for i in range(reference.size):
        # Check the computation results against the reference
        start = time.time()
        result = compiler.execute(kernel, inputs=np.array([inputs[i]]))
        log_dict["execution time (s)"] = log_dict["execution time (s)"] + time.time() - start
        assert np.isclose(result,
                          reference[i]), f"evaluation #{i}: result: {result[0]:16.9f}, reference: {reference[i]:16.8f}"
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=log_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_dict)


def test_all_speakers(models_path, io_path, log_file, max_rows=None):
    if not os.path.isdir(models_path):
        raise Exception("Must provide valid path to models directory")
    if not os.path.isdir(io_path):
        raise Exception("Must provide valid path to IO files directory")
    input_csv = os.path.join(io_path, "input.csv")
    if not os.path.isfile(input_csv):
        raise Exception("No input.csv file in IO files directory")
    inputs = np.genfromtxt(input_csv, delimiter=",", dtype="float64", max_rows=max_rows)
    for path, subdirectories, files in os.walk(models_path):
        print("Found", len(files), "models in folder", path)
        for filename in files:
            spn_name = Path(filename).stem
            print("\t> SPN:", spn_name)
            query = BinaryDeserializer(os.path.join(path, filename)).deserialize_from_file()
            spn = query.root
            reference = np.genfromtxt(fname=os.path.join(io_path, spn_name + ".csv"),
                                      delimiter=",",
                                      dtype="float64",
                                      max_rows=max_rows)
            test_spn(spn=spn, inputs=inputs, reference=reference, log_file=log_file, spn_name=spn_name)


if __name__ == "__main__":
    speaker_models = "/net/celebdil/spn-benchmarks/speaker-identification/models/"
    test_all_speakers(models_path=speaker_models,
                      io_path="/net/celebdil/spn-benchmarks/speaker-identification/io_clean/",
                      max_rows=1000,
                      log_file="normal.log")
    print("DONE")
