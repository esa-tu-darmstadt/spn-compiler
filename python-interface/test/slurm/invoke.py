# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

import fire
import os
import pandas
import re
import subprocess


def toListOrAppend(variable, value):
    if variable is None:
        return [value]
    variable.append(value)
    return variable


def parse_output(output, skip_execution):
    # Outputs that appear only once.
    compile_time_re = re.compile(r"COMPILATION TIME:\s+(.+)\s+ns")
    compile_time = None
    status_re = re.compile(r"STATUS OK")
    status = False
    slp_num_profitable_iterations_re = re.compile(r"profitable SLP iterations:\s+(\d+)")
    num_profitable_iterations = None
    slp_num_ops_pre_re = re.compile(r"#ops before vectorization:\s+(\d+)")
    num_ops_pre = None
    slp_num_ops_post_total_re = re.compile(r"#ops after vectorization \(total\):\s+(\d+)")
    num_ops_post_total = None
    slp_num_ops_post_not_dead_re = re.compile(r"#ops after vectorization \(not dead\):\s+(\d+)")
    num_ops_post_not_dead = None
    slp_total_re = re.compile(r"SLP TOTAL TIME:\s+(\d+)\s+ns")
    slp_total_time = None

    # Outputs that can appear multiple times.
    execution_time_re = re.compile(r"EXECUTION TIME:\s+(\d+)\s+ns")
    num_executions = 0
    execution_time = None
    lifetime_re = re.compile(r"lifetime total in function \(.*task_\d+\):\s+(\d+)")
    lifetimes = None
    op_count_re = re.compile(r"op count: \"(.*)\": (\d+)")
    op_names = None
    op_counts = None
    slp_num_superwords_re = re.compile(r"#superwords in graph \((\d+)\):\s+(\d+)")
    num_superwords = None
    slp_num_arith_ops_re = re.compile(r"#unique arithmetic graph ops \((\d+)\):\s+(\d+)")
    num_arith_ops = None
    slp_cover_re = re.compile(r"% function ops dead \((\d+)\):\s+(.+)%")
    slp_cover = None
    slp_seed_re = re.compile(r"SEED TIME \((\d+)\):\s+(\d+)\s+ns")
    slp_seed_time = None
    slp_graph_re = re.compile(r"GRAPH TIME \((\d+)\):\s+(\d+)\s+ns")
    slp_graph_time = None
    slp_rewrite_re = re.compile(r"PATTERN REWRITE TIME \((\d+)\):\s+(\d+)\s+ns")
    slp_match_rewrite_time = None

    # ==
    for line in output.splitlines():

        if m := compile_time_re.match(line):
            compile_time = int(m.group(1))

        elif status_re.match(line):
            status = True

        elif m := slp_num_profitable_iterations_re.match(line):
            num_profitable_iterations = int(m.group(1))

        if m := slp_num_ops_pre_re.match(line):
            num_ops_pre = int(m.group(1))

        elif m := slp_num_ops_post_total_re.match(line):
            num_ops_post_total = int(m.group(1))

        elif m := slp_num_ops_post_not_dead_re.match(line):
            num_ops_post_not_dead = int(m.group(1))

        elif m := slp_total_re.match(line):
            if slp_total_time is None:
                slp_total_time = 0
            slp_total_time = slp_total_time + int(m.group(1))

        elif m := execution_time_re.match(line):
            if execution_time is None:
                execution_time = 0
            execution_time = execution_time + int(m.group(1))
            num_executions = num_executions + 1

        elif m := lifetime_re.match(line):
            lifetimes = toListOrAppend(lifetimes, int(m.group(1)))

        elif m := op_count_re.match(line):
            op_names = toListOrAppend(op_names, m.group(1))
            op_counts = toListOrAppend(op_counts, m.group(2))

        elif m := slp_num_superwords_re.match(line):
            num_superwords = toListOrAppend(num_superwords, m.group(2))

        elif m := slp_num_arith_ops_re.match(line):
            num_arith_ops = toListOrAppend(num_arith_ops, m.group(2))

        elif m := slp_cover_re.match(line):
            slp_cover = toListOrAppend(slp_cover, m.group(2))

        elif m := slp_seed_re.match(line):
            slp_seed_time = toListOrAppend(slp_seed_time, int(m.group(2)))

        elif m := slp_graph_re.match(line):
            slp_graph_time = toListOrAppend(slp_graph_time, int(m.group(2)))

        elif m := slp_rewrite_re.match(line):
            slp_match_rewrite_time = toListOrAppend(slp_match_rewrite_time, int(m.group(2)))

    if not status or compile_time is None or (not skip_execution and execution_time is None):
        print(f"Status: {status}")
        print(f"Compile time: {compile_time}")
        print(f"Execution time: {execution_time}")
        raise RuntimeError("Time measurement failed")

    data = {"compile time (ns)": compile_time}

    if num_profitable_iterations is not None:
        data["#profitable iterations"] = num_profitable_iterations
    if num_ops_pre is not None:
        data["#lospn ops pre SLP"] = num_ops_pre
    if num_ops_post_total is not None:
        data["#lospn ops post SLP (total)"] = num_ops_post_total
    if num_ops_post_not_dead is not None:
        data["#lospn ops post SLP (not dead)"] = num_ops_post_not_dead
    if slp_total_time is not None:
        data["slp time total (ns)"] = slp_total_time
    if execution_time is not None:
        data["execution time total (ns)"] = execution_time
    if num_executions is not None:
        data["#inferences"] = num_executions
    if lifetimes is not None:
        for i in range(len(lifetimes)):
            data[f"lifetime total in task {i}"] = lifetimes[i]
    if op_names is not None:
        for i in range(len(op_names)):
            data[f"#{op_names[i]}"] = op_counts[i]
    if num_superwords is not None:
        for i in range(len(num_superwords)):
            data[f"#superwords in graph {i}"] = num_superwords[i]
    if num_arith_ops is not None:
        for i in range(len(num_arith_ops)):
            data[f"#unique arithmetic op in graph {i}"] = num_arith_ops[i]
    if slp_cover is not None:
        for i in range(len(slp_cover)):
            data[f"% function ops dead after iteration {i}"] = slp_cover[i]
    if slp_seed_time is not None:
        for i in range(len(slp_seed_time)):
            data[f"slp seed time {i} (ns)"] = slp_seed_time[i]
    if slp_graph_time is not None:
        for i in range(len(slp_graph_time)):
            data[f"slp graph time {i} (ns)"] = slp_graph_time[i]
    if slp_match_rewrite_time is not None:
        for i in range(len(slp_match_rewrite_time)):
            data[f"slp pattern match/rewrite time {i} (ns)"] = slp_match_rewrite_time[i]

    return data


def invokeCompileAndExecute(logDir, modelName, modelFile, inputFile, referenceFile, vectorize, vecLib, shuffle,
                            maxAttempts=None, maxSuccessfulIterations=None, maxNodeSize=None, maxLookAhead=None,
                            reorderInstructionsDFS=None, allowDuplicateElements=None, allowTopologicalMixing=None,
                            useXorChains=None, maxTaskSize=None, skipExecution=None, kernelDir=None):
    command = ["python3", os.path.join(os.path.dirname(os.path.realpath(__file__)), "cpuExecutionSlurm.py")]
    # model name and model file
    command.extend(("--name", modelName, "--spn_file", modelFile))
    # input and reference paths
    command.extend(("--input_data", inputFile, "--reference_data", referenceFile))
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
    if useXorChains is not None:
        command.extend(("--useXorChains", str(useXorChains)))
    if maxTaskSize is not None:
        print("WARNING: output parsing does not fully work for partitioned tasks")
        command.extend(("--maxTaskSize", str(maxTaskSize)))
    if skipExecution is not None and skipExecution:
        command.extend(("--skipExecution", str(skipExecution)))
    if kernelDir is not None:
        command.extend(("--kernel_dir", kernelDir))

    run_result = subprocess.run(command, capture_output=True, text=True)
    if run_result.returncode == 0:
        parsed_data = parse_output(run_result.stdout, skipExecution)
        data = {"Name": modelName}
        data.update(parsed_data)
        if not os.path.isdir(logDir):
            os.mkdir(logDir)
        log_file = os.path.join(logDir, "data.csv")
        if not os.path.exists(log_file):
            df = pandas.DataFrame(data, index=[0])
            df.to_csv(log_file, index=False)
        else:
            df = pandas.read_csv(log_file)
            for column in [key for key in data.keys() if key not in df.columns]:
                df.insert(len(df.columns), column=column, value="")
            df = df.append(data, ignore_index=True)
            df.to_csv(log_file, index=False)
    else:
        print(f"Compilation and execution of {modelName} failed with error code {run_result.returncode}")
        print(f"Command was: {command}")
        print(run_result.stdout)
        print(run_result.stderr)


if __name__ == '__main__':
    fire.Fire(invokeCompileAndExecute)
