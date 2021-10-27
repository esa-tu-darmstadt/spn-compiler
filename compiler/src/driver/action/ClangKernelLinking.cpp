//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include <util/Command.h>
#include "ClangKernelLinking.h"
#include "toolchain/MLIRToolchain.h"

using namespace spnc;

spnc::ExecutionResult ClangKernelLinking::executeStep(ObjectFile* objectFile, SharedObject* sharedObject) {
  // Invoke clang as external command.
  std::vector<std::string> command;
  command.emplace_back("clang");
  command.emplace_back("-shared");
  command.emplace_back("-fPIC");
  command.emplace_back("-O3");
  command.emplace_back("-o");
  command.push_back(sharedObject->fileName());
  command.push_back(objectFile->fileName());
  auto* libraryInfo = getContext()->get<LibraryInfo>();
  for (auto& lib: libraryInfo->libraries()) {
    command.push_back("-l" + lib);
  }
  for (auto& path: libraryInfo->searchPaths()) {
    command.push_back("-L " + path);
  }
  Command::executeExternalCommand(command);
  auto* kernelInfo = getContext()->get<KernelInfo>();
  kernel = std::make_unique<Kernel>(sharedObject->fileName(), kernelInfo->kernelName,
                                    kernelInfo->queryType, kernelInfo->target, kernelInfo->batchSize,
                                    kernelInfo->numFeatures, kernelInfo->bytesPerFeature,
                                    kernelInfo->numResults, kernelInfo->bytesPerResult,
                                    kernelInfo->dtype);
  outFile = sharedObject;
  return success();
}

Kernel* ClangKernelLinking::result() {
  return kernel.get();
}

std::string ClangKernelLinking::stepName = "kernel-linking";