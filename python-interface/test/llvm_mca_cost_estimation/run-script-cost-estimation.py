# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

import csv
import fire
import os
import re
import subprocess


def parse_output(output):
    """
    Example data found in output:

    [...]

    Iterations:        100

    Instructions:      793700

    Total Cycles:      329044

    Total uOps:        1286200


    Dispatch Width:    6

    uOps Per Cycle:    3.91

    IPC:               2.41

    Block RThroughput: 2963.0

    [...]

    STATUS OK
    """
    iterations_re = re.compile(r"Iterations:\s+(\d+)")
    instructions_re = re.compile(r"Instructions:\s+(\d+)")
    total_cycles_re = re.compile(r"Total Cycles:\s+(\d+)")
    total_uops_re = re.compile(r"Total uOps:\s+(\d+)")
    dispatch_width_re = re.compile(r"Dispatch Width:\s+(\d+)")
    uops_per_cycle_re = re.compile(r"uOps Per Cycle:\s+([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)")
    ipc_re = re.compile(r"IPC:\s+([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)")
    block_rthroughput_re = re.compile(r"Block RThroughput:\s+([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)")
    status_re = re.compile(r"STATUS OK")
    iterations = None
    instructions = None
    total_cycles = None
    total_uops = None
    dispatch_width = None
    uops_per_cycle = None
    ipc = None
    block_rthroughput = None
    status = False
    for line in output.splitlines():
        if m := iterations_re.match(line):
            iterations = int(m.group(1))
        elif m := instructions_re.match(line):
            instructions = int(m.group(1))
        elif m := total_cycles_re.match(line):
            total_cycles = int(m.group(1))
        elif m := total_uops_re.match(line):
            total_uops = int(m.group(1))
        elif m := dispatch_width_re.match(line):
            dispatch_width = int(m.group(1))
        elif m := uops_per_cycle_re.match(line):
            uops_per_cycle = float(m.group(1))
        elif m := ipc_re.match(line):
            ipc = float(m.group(1))
        elif m := block_rthroughput_re.match(line):
            block_rthroughput = float(m.group(1))
        elif status_re.match(line):
            status = True
    if not status:
        raise RuntimeError("Cost estimation failed")
    return {"Iterations": iterations,
            "Instructions": instructions,
            "Total Cycles": total_cycles,
            "Total uOps": total_uops,
            "Dispatch Width": dispatch_width,
            "uOps Per Cycle": uops_per_cycle,
            "IPC": ipc,
            "Block RThroughput": block_rthroughput}


def parse_cost_output(file):
    """
    Example cost data found in file:

    SLP Iteration: 0

    Estimated Cost: 6125.000000
    """
    iteration_re = re.compile(r"SLP Iteration:\s+(\d+)")
    estimated_cost_re = re.compile(r"Estimated Cost:\s+([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)")
    iteration = None
    estimated_cost = None
    with open(file, "r") as cost_file:
        for line in cost_file.readlines():
            if m := iteration_re.match(line):
                iteration = int(m.group(1))
            elif m := estimated_cost_re.match(line):
                estimated_cost = float(m.group(1))
    if iteration is None or estimated_cost is None:
        raise RuntimeError("Cost estimation parsing failed")
    return {"SLP Iteration:": iteration, "Estimated Cost:": estimated_cost}


def invokeCompileAndExecute(model, vecLib, shuffle, mca_iterations):
    command = ["python3", os.path.join(os.path.dirname(os.path.realpath(__file__)), "cpu-execution-cost-estimation.py")]
    command.extend((model[0], model[1]))
    command.extend((str(True), str(vecLib), str(shuffle), str(mca_iterations)))
    run_result = subprocess.run(command, capture_output=True, text=True)
    if run_result.returncode == 0:
        return parse_output(run_result.stdout)
    else:
        print(f"Compilation and execution of {model[0]} failed")
        print(run_result.stdout)
        print(run_result.stderr)
    return None


def traverseSpeakers(modelDir: str, logDir: str, vecLib: bool, shuffle: bool, mca_iterations: int):
    models = []
    for subdir, dirs, files in os.walk(modelDir):
        for file in files:
            baseName = os.path.basename(file)
            extension = os.path.splitext(baseName)[-1].lower()
            modelName = os.path.splitext(baseName)[0]
            if extension == ".bin":
                modelFile = os.path.join(subdir, file)
                models.append((modelName, modelFile))
    print(f"Number of models found: {len(models)}")

    # If there exists a cost analysis file already, delete it.
    slp_cost_file = "costAnalysis.log"
    if os.path.exists(slp_cost_file):
        os.remove(slp_cost_file)

    for m in models:
        print(f"Current model: {m[0]}")
        llvm_cost = invokeCompileAndExecute(m, vecLib, shuffle, mca_iterations)

        # Check that my own cost analysis file exists.
        slp_cost_file = "costAnalysis.log"
        if not os.path.isfile(slp_cost_file):
            raise RuntimeError("Cost file not found. Did you forget enabling the cost analysis output in SPNC?")

        estimated_cost = parse_cost_output(slp_cost_file)

        data = {"Name:": m[0]}
        data.update(llvm_cost)
        data.update(estimated_cost)

        log_file_all = os.path.join(logDir, "costAnalysis_3.csv")
        file_exists = os.path.isfile(log_file_all)
        if not os.path.isdir(logDir):
            os.mkdir(logDir)
        with open(log_file_all, 'a') as log_file:
            writer = csv.DictWriter(log_file, delimiter=",", fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)


if __name__ == '__main__':
    fire.Fire(traverseSpeakers)
