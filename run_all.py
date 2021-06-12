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


def read_csv(csv_file, delimiter=",", dtype="float64", max_rows=None):
    if not os.path.isfile(csv_file):
        raise Exception("CSV file does not exist")
    data = np.genfromtxt(fname=csv_file, delimiter=delimiter, dtype=dtype, max_rows=max_rows)
    if data.ndim < 2:
        data = np.atleast_2d(data)
    return data


def test_spn(spn, inputs, references, spn_name, log_file):
    # Things to log.
    log_dict = {
        "name": spn_name,
        "execution time (s)": 0,
        "compile time (s)": 0,
        "num_evaluations": references.size
    }
    # Compile the kernel.
    compiler = CPUCompiler(vectorize=False, computeInLogSpace=True)
    start = time.time()
    kernel = compiler.compile_ll(spn=spn, batchSize=1, supportMarginal=False)
    log_dict["compile time (s)"] = time.time() - start
    # Execute the compiled kernel.
    for i in range(references.size):
        # Check the computation results against the reference
        reference = references[i]
        start = time.time()
        result = compiler.execute(kernel, inputs=np.atleast_2d(inputs[i]))
        log_dict["execution time (s)"] = log_dict["execution time (s)"] + time.time() - start
        assert np.isclose(result, reference), f"evaluation #{i}: result: {result:16.9f}, reference: {reference:16.8f}"
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=log_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_dict)


def test_speakers(models_path, io_path, log_file, max_rows=None):
    if not os.path.isdir(models_path):
        raise Exception("Must provide valid path to models directory")
    if not os.path.isdir(io_path):
        raise Exception("Must provide valid path to IO files directory")
    inputs = read_csv(os.path.join(io_path, "input.csv"),
                      delimiter=",",
                      dtype="float64",
                      max_rows=max_rows)
    for path, subdirectories, files in os.walk(models_path):
        print("Found", len(files), "models in folder", path)
        for filename in files:
            spn_name = Path(filename).stem
            print("\t> SPN:", spn_name)
            query = BinaryDeserializer(os.path.join(path, filename)).deserialize_from_file()
            spn = query.root
            references = read_csv(os.path.join(io_path, spn_name + ".csv"),
                                  delimiter=",",
                                  dtype="float64",
                                  max_rows=max_rows)
            references = references.reshape(references.size)
            test_spn(spn=spn, inputs=inputs, references=references, log_file=log_file, spn_name=spn_name)


def test_all_speakers(log_prefix, max_rows=None):
    speaker_models = "/net/celebdil/spn-benchmarks/speaker-identification/models/"
    test_speakers(models_path=speaker_models,
                  io_path="/net/celebdil/spn-benchmarks/speaker-identification/io_clean/",
                  max_rows=max_rows,
                  log_file=log_prefix + "_clean.log")
    test_speakers(models_path=speaker_models,
                  io_path="/net/celebdil/spn-benchmarks/speaker-identification/io_noisy_marg_0_bounds_0/",
                  max_rows=max_rows,
                  log_file=log_prefix + "_io_noisy_marg_0_bounds_0.log")
    test_speakers(models_path=speaker_models,
                  io_path="/net/celebdil/spn-benchmarks/speaker-identification/io_noisy_marg_1_bounds_0/",
                  max_rows=max_rows,
                  log_file=log_prefix + "_io_noisy_marg_1_bounds_0.log")
    test_speakers(models_path=speaker_models,
                  io_path="/net/celebdil/spn-benchmarks/speaker-identification/io_noisy_marg_1_bounds_1/",
                  max_rows=max_rows,
                  log_file=log_prefix + "_io_noisy_marg_1_bounds_1.log")


if __name__ == "__main__":
    test_all_speakers(log_prefix="speakers", max_rows=10)
    print("DONE")
