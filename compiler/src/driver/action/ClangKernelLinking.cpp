//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <util/Command.h>
#include "ClangKernelLinking.h"

using namespace spnc;

ClangKernelLinking::ClangKernelLinking(ActionWithOutput<ObjectFile>& _input,
                                       SharedObject outputFile, std::shared_ptr<KernelInfo> info)
    : ActionSingleInput<ObjectFile, Kernel>(_input), outFile{std::move(outputFile)}, kernelInfo{std::move(info)} {}

Kernel& ClangKernelLinking::execute() {
  if (!cached) {
    // Invoke clang as external command.
    std::vector<std::string> command;
    command.emplace_back("clang");
    command.emplace_back("-shared");
    command.emplace_back("-fPIC");
    command.emplace_back("-O3");
    command.emplace_back("-o");
    command.push_back(outFile.fileName());
    command.push_back(input.execute().fileName());
    Command::executeExternalCommand(command);
    kernel = std::make_unique<Kernel>(outFile.fileName(), kernelInfo->kernelName,
                                      kernelInfo->queryType, kernelInfo->batchSize,
                                      kernelInfo->numFeatures, kernelInfo->bytesPerFeature,
                                      kernelInfo->numResults, kernelInfo->bytesPerResult,
                                      kernelInfo->dtype);
    cached = true;
  }
  return *kernel;
}

