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


def traverseModels(speakersDir: str, speakersDataDir: str, ratspnDir: str, ratspnDataDir: str, arrayDir: str,
                   arrayFile: str, vectorize: bool, vecLib: str, shuffle: bool, maxAttempts=None,
                   maxSuccessfulIterations=None, maxNodeSize=None, maxLookAhead=None, reorderInstructionsDFS=None,
                   allowDuplicateElements=None, allowTopologicalMixing=None, useXorChains=None, taskPartitions=None,
                   skipExecution=None, kernelDir=None):
    models = []
    models.extend(get_speakers(speakersDir, speakersDataDir))
    models.extend(get_ratspns(ratspnDir, ratspnDataDir))

    # Sort models s.t. smaller ones are executed first
    models = sorted(models, key=lambda m: os.path.getsize(m[1]))

    counter = 0
    for m in [models[-13], models[-12], models[-11], models[-10], models[-9], models[-8], models[-7], models[-6],
              models[-5], models[-4], models[-3], models[-2], models[-1]]:
        print(f"Skipping model {m} because of traversal limit in words problems.")
        counter = counter + 1

    if not os.path.isdir(arrayDir):
        os.makedirs(arrayDir)

    if kernelDir is not None and not os.path.isdir(kernelDir):
        os.makedirs(kernelDir)

    with open(os.path.join(arrayDir, arrayFile), 'w') as file:
        for m in models[:-13]:
            arguments = [
                "--modelName", m[0],
                "--modelFile", m[1],
                "--inputFile", m[2],
                "--referenceFile", m[3],
                "--vectorize", str(vectorize),
                "--vecLib", str(vecLib),
                "--shuffle", str(shuffle)
            ]
            if maxAttempts is not None:
                arguments.extend(("--maxAttempts", str(maxAttempts)))
            if maxSuccessfulIterations is not None:
                arguments.extend(("--maxSuccessfulIterations", str(maxSuccessfulIterations)))
            if maxNodeSize is not None:
                arguments.extend(("--maxNodeSize", str(maxNodeSize)))
            if maxLookAhead is not None:
                arguments.extend(("--maxLookAhead", str(maxLookAhead)))
            if reorderInstructionsDFS is not None:
                arguments.extend(("--reorderInstructionsDFS", str(reorderInstructionsDFS)))
            if allowDuplicateElements is not None:
                arguments.extend(("--allowDuplicateElements", str(allowDuplicateElements)))
            if allowTopologicalMixing is not None:
                arguments.extend(("--allowTopologicalMixing", str(allowTopologicalMixing)))
            if useXorChains is not None:
                arguments.extend(("--useXorChains", str(useXorChains)))
            if taskPartitions is not None and taskPartitions > 1:
                spn_sizes = pandas.read_csv("lo_spn_sizes_all.csv", index_col="Name")
                max_task_size = spn_sizes.loc[m[0], "#lospn ops"] // taskPartitions
                arguments.extend(("--maxTaskSize", str(max_task_size)))
            if skipExecution is not None:
                arguments.extend(("--skipExecution", str(skipExecution)))
            if kernelDir is not None:
                arguments.extend(("--kernelDir", kernelDir))
            file.write(' '.join(arguments))
            file.write('\n')


if __name__ == '__main__':
    fire.Fire(traverseModels)
