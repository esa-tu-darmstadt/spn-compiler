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


def parse_output(output, vectorize, expected_iterations=1):
    compile_time_re = re.compile(r"COMPILATION TIME:\s+(.+)\s+ns")
    compile_time = None
    execution_time_re = re.compile(r"EXECUTION TIME:\s+(\d+)\s+ns")
    num_executions = 0
    execution_time = None
    status_re = re.compile(r"STATUS OK")
    status = False

    # Optional SLP output if SLP vectorization took place.
    slp_num_ops_pre_re = re.compile(r"#ops before vectorization:\s+(\d+)")
    num_ops_pre = None
    slp_num_ops_post_total_re = re.compile(r"#ops after vectorization \(total\):\s+(\d+)")
    num_ops_post_total = None
    slp_num_ops_post_not_dead_re = re.compile(r"#ops after vectorization \(not dead\):\s+(\d+)")
    num_ops_post_not_dead = None
    slp_num_profitable_iterations_re = re.compile(r"profitable SLP iterations:\s+(\d+)")
    num_profitable_iterations = None
    slp_num_superwords_re = re.compile(r"#superwords in graph \((\d+)\):\s+(\d+)")
    num_superwords = [0] * expected_iterations
    slp_num_arith_ops_re = re.compile(r"#unique arithmetic graph ops \((\d+)\):\s+(\d+)")
    num_arith_ops = [0] * expected_iterations
    slp_cover_re = re.compile(r"% function ops dead \((\d+)\):\s+(.+)%")
    slp_cover = [0] * expected_iterations

    slp_seed_re = re.compile(r"SEED TIME \((\d+)\):\s+(\d+)\s+ns")
    slp_seed_time = [0] * expected_iterations
    slp_graph_re = re.compile(r"GRAPH TIME \((\d+)\):\s+(\d+)\s+ns")
    slp_graph_time = [0] * expected_iterations
    slp_rewrite_re = re.compile(r"PATTERN REWRITE TIME \((\d+)\):\s+(\d+)\s+ns")
    slp_match_rewrite_time = [0] * expected_iterations
    slp_total_re = re.compile(r"SLP TOTAL TIME:\s+(\d+)\s+ns")
    slp_total_time = 0

    # ==
    for line in output.splitlines():

        if m := compile_time_re.match(line):
            compile_time = int(m.group(1))

        elif m := execution_time_re.match(line):
            if execution_time is None:
                execution_time = 0
            execution_time = execution_time + int(m.group(1))
            num_executions = num_executions + 1

        elif status_re.match(line):
            status = True

        elif vectorize:

            if m := slp_num_ops_pre_re.match(line):
                num_ops_pre = int(m.group(1))

            elif m := slp_num_ops_post_total_re.match(line):
                num_ops_post_total = int(m.group(1))

            elif m := slp_num_ops_post_not_dead_re.match(line):
                num_ops_post_not_dead = int(m.group(1))

            elif m := slp_num_profitable_iterations_re.match(line):
                num_profitable_iterations = int(m.group(1))

            elif m := slp_num_superwords_re.match(line):
                num_superwords[int(m.group(1))] = m.group(2)

            elif m := slp_num_arith_ops_re.match(line):
                num_arith_ops[int(m.group(1))] = m.group(2)

            elif m := slp_cover_re.match(line):
                slp_cover[int(m.group(1))] = m.group(2)

            elif m := slp_seed_re.match(line):
                slp_seed_time[int(m.group(1))] = int(m.group(2))

            elif m := slp_graph_re.match(line):
                slp_graph_time[int(m.group(1))] = int(m.group(2))

            elif m := slp_rewrite_re.match(line):
                slp_match_rewrite_time[int(m.group(1))] = int(m.group(2))

            elif m := slp_total_re.match(line):
                slp_total_time = slp_total_time + int(m.group(1))

    if not status or compile_time is None or execution_time is None or (vectorize and num_ops_post_total is None):
        print(f"Status: {status}")
        print(f"Compile time: {compile_time}")
        print(f"Execution time: {execution_time}")
        raise RuntimeError("Time measurement failed")

    data = {
        "compile time (ns)": compile_time,
        "#lospn ops pre SLP": num_ops_pre,
        "#lospn ops post SLP (total)": num_ops_post_total,
        "#lospn ops post SLP (not dead)": num_ops_post_not_dead,
        "#profitable iterations": num_profitable_iterations,
        "execution time total (ns)": execution_time,
        "#inferences": num_executions,
        "slp time total (ns)": slp_total_time
    }
    for i in range(expected_iterations):
        data[f"#superwords in graph {i}"] = num_superwords[i]
        data[f"#unique arithmetic op in graph {i}"] = num_arith_ops[i]
        data[f"% function ops dead after iteration {i}"] = slp_cover[i]
        data[f"slp seed time {i} (ns)"] = slp_seed_time[i]
        data[f"slp graph time {i} (ns)"] = slp_graph_time[i]
        data[f"slp pattern match/rewrite time {i} (ns)"] = slp_match_rewrite_time[i]
    return data


def invokeCompileAndExecute(logDir, modelName, modelFile, inputFile, referenceFile, kernelDir, removeKernel,
                            vectorize, vecLib, shuffle, maxAttempts=None, maxSuccessfulIterations=None,
                            maxNodeSize=None, maxLookAhead=None, reorderInstructionsDFS=None,
                            allowDuplicateElements=None, allowTopologicalMixing=None, maxTaskSize=None):
    command = ["python3", os.path.join(os.path.dirname(os.path.realpath(__file__)), "cpuExecutionSlurm.py")]
    # model name and model file
    command.extend(("--name", modelName, "--spn_file", modelFile))
    # input and reference paths
    command.extend(("--input_data", inputFile, "--reference_data", referenceFile))
    command.extend(("--kernel_dir", kernelDir, "--remove_kernel", str(removeKernel)))
    command.extend(("--vectorize", str(vectorize), "--vectorLibrary", str(vecLib), "--shuffle", str(shuffle)))

    if maxAttempts is not None:
        command.extend(("--maxAttempts", str(maxAttempts)))
    if maxSuccessfulIterations is not None:
        command.extend(("--maxSuccessfulIterations", str(maxSuccessfulIterations)))
    if maxNodeSize is not None:
        command.extend(("--maxNodeSize", str(maxNodeSize)))
    if maxLookAhead is not None:
        command.extend(("--maxLookAhead", str(maxLookAhead)))
    if reorderInstructionsDFS is not None:
        command.extend(("--reorderInstructionsDFS", str(reorderInstructionsDFS)))
    if allowDuplicateElements is not None:
        command.extend(("--allowDuplicateElements", str(allowDuplicateElements)))
    if allowTopologicalMixing is not None:
        command.extend(("--allowTopologicalMixing", str(allowTopologicalMixing)))
    if maxTaskSize is not None:
        command.extend(("--maxTaskSize", str(maxTaskSize)))

    run_result = subprocess.run(command, capture_output=True, text=True)
    if run_result.returncode == 0:
        if maxSuccessfulIterations is not None:
            parsed_data = parse_output(run_result.stdout, vectorize, maxSuccessfulIterations)
        else:
            parsed_data = parse_output(run_result.stdout, vectorize)
        data = {"Name": modelName}
        data.update(parsed_data)
        log_file_all = os.path.join(logDir, "data.csv")
        file_exists = os.path.isfile(log_file_all)
        if not os.path.isdir(logDir):
            os.mkdir(logDir)
        with open(log_file_all, 'a') as log_file:
            writer = csv.DictWriter(log_file, delimiter=",", fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
    else:
        print(f"Compilation and execution of {modelName} failed with error code {run_result.returncode}")
        print(f"Command was: {command}")
        print(run_result.stdout)
        print(run_result.stderr)


if __name__ == '__main__':
    fire.Fire(invokeCompileAndExecute)
