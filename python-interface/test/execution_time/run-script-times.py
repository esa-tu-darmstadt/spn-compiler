#!/usr/bin/env python3
import csv
import fire
import os
import re
import subprocess


def parse_output(output):
    """
    Example data found in output:

    [...]
    COMPILATION TIME: 12345 ns

    [...]

    EXECUTION TIME: 5253 ns

    [...]

    STATUS OK
    """
    compile_time_re = re.compile(r"COMPILATION TIME:\s+(\d+)\s+ns")
    time_re = re.compile(r"EXECUTION TIME:\s+(\d+)\s+ns")
    status_re = re.compile(r"STATUS OK")
    compile_time = None
    time = None
    num_executions = 0
    status = False
    for line in output.splitlines():
        if m := compile_time_re.match(line):
            compile_time = int(m.group(1))
        elif m := time_re.match(line):
            if time is None:
                time = 0
            time = time + int(m.group(1))
            num_executions = num_executions + 1
        elif status_re.match(line):
            status = True
    if not status or compile_time is None or time is None:
        raise RuntimeError("Time measurement failed")
    return {"compile time (ns)": compile_time, "execution time total (ns)": time, "#inferences": num_executions}


def invokeCompileAndExecute(model, vectorize, vecLib, shuffle):
    command = ["python3", os.path.join(os.path.dirname(os.path.realpath(__file__)), "cpu-execution-times.py")]
    # model name and model file
    command.extend((model[0], model[1]))
    # input and reference paths
    command.extend((model[2], model[3]))
    command.extend((str(vectorize), str(vecLib), str(shuffle)))
    run_result = subprocess.run(command, capture_output=True, text=True)
    if run_result.returncode == 0:
        return parse_output(run_result.stdout)
    else:
        print(f"Compilation and execution of {model[0]} failed")
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
                   vectorize: bool, vecLib: bool, shuffle: bool):
    models = []
    models.extend(get_speakers(speakersDir, speakersDataDir))
    models.extend(get_ratspns(ratspnDir, ratspnDataDir))

    # Sort models s.t. smaller ones are executed first
    models = sorted(models, key=lambda m: os.path.getsize(m[1]))
    for m in [models[-13], models[-12], models[-11], models[-10], models[-9], models[-8], models[-7], models[-6],
              models[-5], models[-4], models[-3], models[-2], models[-1]]:
        print(f"Skipping model {m} because of traversal limit in words problems.")
        print(f"\tFile size of model: {os.path.getsize(m[1])}")

    for m in models:
        print(f"Current model: {m[0]}")
        times = invokeCompileAndExecute(m, vectorize, vecLib, shuffle)
        data = {"Name": m[0]}
        data.update(times)

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
