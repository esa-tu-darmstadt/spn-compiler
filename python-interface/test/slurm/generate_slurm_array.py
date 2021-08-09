# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

import fire
import os


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
                   arrayFile: str, kernelDir: str, removeKernels: bool, vectorize: bool, vecLib: str, shuffle: bool,
                   maxAttempts=None, maxSuccessfulIterations=None, maxNodeSize=None, maxLookAhead=None,
                   reorderInstructionsDFS=None, allowDuplicateElements=None, allowTopologicalMixing=None):
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

    if not os.path.isdir(kernelDir):
        os.makedirs(kernelDir)

    with open(os.path.join(arrayDir, arrayFile), 'w') as file:
        optionals = [str(maxAttempts), str(maxSuccessfulIterations), str(maxNodeSize), str(maxLookAhead),
                     str(reorderInstructionsDFS), str(allowDuplicateElements), str(allowTopologicalMixing)]
        optionals_string = ' '.join(opt for opt in optionals if opt != "None")
        for m in models[:-13]:
            file.write(
                f"{m[0]} {m[1]} {m[2]} {m[3]} {kernelDir} {removeKernels} {vectorize} {vecLib} {shuffle} {optionals_string}\n"
            )


if __name__ == '__main__':
    fire.Fire(traverseModels)
