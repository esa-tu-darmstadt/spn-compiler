#!/usr/bin/env python3
import csv
import fire
import os
import re
import subprocess


def parse_output(output, expected_iterations=1):
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
    slp_num_ops_post_re = re.compile(r"#ops after vectorization:\s+(\d+)")
    num_ops_post = None
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
            continue

        if m := slp_num_ops_pre_re.match(line):
            num_ops_pre = int(m.group(1))
            continue

        if m := slp_num_ops_post_re.match(line):
            num_ops_post = int(m.group(1))
            continue

        if m := slp_num_profitable_iterations_re.match(line):
            num_profitable_iterations = int(m.group(1))
            continue

        if m := slp_num_superwords_re.match(line):
            num_superwords[int(m.group(1))] = m.group(2)
            continue

        if m := slp_num_arith_ops_re.match(line):
            num_arith_ops[int(m.group(1))] = m.group(2)
            continue

        if m := slp_cover_re.match(line):
            slp_cover[int(m.group(1))] = m.group(2)
            continue

        if m := slp_seed_re.match(line):
            slp_seed_time[int(m.group(1))] = int(m.group(2))
            continue

        if m := slp_graph_re.match(line):
            slp_graph_time[int(m.group(1))] = int(m.group(2))
            continue

        if m := slp_rewrite_re.match(line):
            slp_match_rewrite_time[int(m.group(1))] = int(m.group(2))
            continue

        if m := slp_total_re.match(line):
            slp_total_time = slp_total_time + int(m.group(1))
            continue

        if m := execution_time_re.match(line):
            if execution_time is None:
                execution_time = 0
            execution_time = execution_time + int(m.group(1))
            num_executions = num_executions + 1
            continue

        if status_re.match(line):
            status = True
            continue

    if not status or compile_time is None or execution_time is None:
        print(f"Status: {status}")
        print(f"Compile time: {compile_time}")
        print(f"Execution time: {execution_time}")
        raise RuntimeError("Time measurement failed")

    data = {
        "compile time (ns)": compile_time,
        "#lospn ops pre SLP": num_ops_pre,
        "#lospn ops post SLP": num_ops_post,
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


def invokeCompileAndExecute(model, vectorize, vecLib, shuffle, maxAttempts=None, maxSuccessfulIterations=None,
                            maxNodeSize=None, maxLookAhead=None, reorderInstructionsDFS=None,
                            allowDuplicateElements=None, allowTopologicalMixing=None):
    command = ["python3", os.path.join(os.path.dirname(os.path.realpath(__file__)), "cpuExecutionTimes.py")]
    # model name and model file
    command.extend((model[0], model[1]))
    # input and reference paths
    command.extend((model[2], model[3]))
    command.extend((str(vectorize), str(vecLib), str(shuffle)))

    if maxAttempts is not None:
        command.append(str((maxAttempts)))
    if maxSuccessfulIterations is not None:
        command.append(str((maxSuccessfulIterations)))
    if maxNodeSize is not None:
        command.append(str((maxNodeSize)))
    if maxLookAhead is not None:
        command.append(str((maxLookAhead)))
    if reorderInstructionsDFS is not None:
        command.append(str((reorderInstructionsDFS)))
    if allowDuplicateElements is not None:
        command.append(str((allowDuplicateElements)))
    if allowTopologicalMixing is not None:
        command.append(str((allowTopologicalMixing)))

    run_result = subprocess.run(command, capture_output=True, text=True)
    if run_result.returncode == 0:
        return parse_output(run_result.stdout, maxSuccessfulIterations)
    else:
        print(f"Compilation and execution of {model[0]} failed with error code {run_result.returncode}")
        print(run_result.stdout)
        print(run_result.stderr)
    return None


def getIOFolder(ratspnModel: str):
    if "fashion_mnist" in ratspnModel:
        return "fashion_mnist"
    elif "pumsb_star" in ratspnModel:
        return "pumsb_star"
    return ratspnModel[0:ratspnModel.index("_")]


def get_ratspns(ratspnDir: str, dataDir: str):
    models = []
    for subdir, dirs, files in os.walk(ratspnDir):
        for file in files:
            baseName = os.path.basename(file)
            extension = os.path.splitext(baseName)[-1].lower()
            modelName = subdir.replace(ratspnDir, '') + "_" + os.path.splitext(baseName)[0]
            if extension == ".bin":
                inputFile = os.path.join(dataDir, getIOFolder(modelName), "input.csv")
                if not os.path.isfile(inputFile):
                    print(f"Did not find input data for {modelName}, skipping")
                referenceFile = os.path.join(dataDir, getIOFolder(modelName), f"{modelName}.csv")
                if not os.path.isfile(referenceFile):
                    print(f"Did not find reference data for {modelName}, skipping")
                else:
                    modelFile = os.path.join(subdir, file)
                    models.append((modelName, modelFile, inputFile, referenceFile))
    print(f"Number of RAT-SPN models found: {len(models)}")
    return models


def get_speakers(speakersDir: str, dataDir: str):
    models = []
    for subdir, dirs, files in os.walk(speakersDir):
        for file in files:
            baseName = os.path.basename(file)
            extension = os.path.splitext(baseName)[-1].lower()
            modelName = os.path.splitext(baseName)[0]
            if extension == ".bin":
                inputFile = os.path.join(dataDir, "input.csv")
                if not os.path.isfile(inputFile):
                    print(f"Did not find input data for {modelName}, skipping")
                referenceFile = os.path.join(dataDir, f"{modelName}.csv")
                if not os.path.isfile(referenceFile):
                    print(f"Did not find reference data for {modelName}, skipping")
                else:
                    modelFile = os.path.join(subdir, file)
                    models.append((modelName, modelFile, inputFile, referenceFile))
    print(f"Number of speaker models found: {len(models)}")
    return models


def traverseModels(speakersDir: str, speakersDataDir: str, ratspnDir: str, ratspnDataDir: str, logDir: str,
                   vectorize: bool, vecLib: bool, shuffle: bool, maxAttempts=None, maxSuccessfulIterations=None,
                   maxNodeSize=None, maxLookAhead=None, reorderInstructionsDFS=None, allowDuplicateElements=None,
                   allowTopologicalMixing=None):
    models = []
    models.extend(get_speakers(speakersDir, speakersDataDir))
    models.extend(get_ratspns(ratspnDir, ratspnDataDir))

    # Sort models s.t. smaller ones are executed first
    models = sorted(models, key=lambda m: os.path.getsize(m[1]))

    counter = 0
    for m in [models[-13], models[-12], models[-11], models[-10], models[-9], models[-8], models[-7], models[-6],
              models[-5], models[-4], models[-3], models[-2], models[-1]]:
        print(f"Skipping model {m} because of traversal limit in words problems.")
        print(f"\tFile size of model: {os.path.getsize(m[1])} bytes")
        counter = counter + 1

    for m in models[:-13]:
        counter = counter + 1
        print(f"Current model ({counter}/{len(models)}): {m[0]} ({vectorize})")
        data = {"Name": m[0]}
        data.update(invokeCompileAndExecute(m, vectorize, vecLib, shuffle, maxAttempts, maxSuccessfulIterations,
                                            maxNodeSize, maxLookAhead, reorderInstructionsDFS,
                                            allowDuplicateElements, allowTopologicalMixing))
        log_file_all = os.path.join(logDir, "times.csv")
        file_exists = os.path.isfile(log_file_all)
        if not os.path.isdir(logDir):
            os.mkdir(logDir)
        with open(log_file_all, 'a') as log_file:
            writer = csv.DictWriter(log_file, delimiter=",", fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)


if __name__ == '__main__':
    fire.Fire(traverseModels)
