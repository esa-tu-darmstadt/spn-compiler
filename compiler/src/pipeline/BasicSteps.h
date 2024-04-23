//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_COMPILER_INCLUDE_DRIVER_PIPELINE_BASICSTEPS_H
#define SPNC_COMPILER_INCLUDE_DRIVER_PIPELINE_BASICSTEPS_H

#include "PipelineStep.h"
#include "spdlog/fmt/fmt.h"
#include "util/FileSystem.h"
#if __cplusplus < 201700
#include <experimental/filesystem>
#else
#include <filesystem>
#endif

namespace spnc {

/// Step to locate an existing file as input to the compilation pipeline.
/// \tparam FT File type of the file.
template <FileType FT>
class LocateFile : public StepBase, public StepWithResult<File<FT>> {

public:
  /// Constructor.
  /// \param fileName File name of the existing file.
  explicit LocateFile(std::string fileName)
      : StepBase(fmt::format("locate-file ({})", fileName)),
        fName{std::move(fileName)}, file{"tmp", false} {}

  ExecutionResult execute() override {
    // Check for existence of the file.
#if __cplusplus < 201700
    auto isPresent = std::experimental::filesystem::exists(fName);
#else
    auto isPresent = std::filesystem::exists(fName);
#endif
    if (!isPresent) {
      return failure("Failed to located file {}", fName);
    }
    file = File<FT>{fName, false};
    valid = true;
    return success();
  }

  File<FT> *result() override { return &file; }

private:
  std::string fName;

  File<FT> file;

  bool valid = false;
};

/// Step to create a new temporary file as part of the compilation pipeline.
/// \tparam FT File type of the file to be created.
template <FileType FT>
class CreateTmpFile : public StepBase, public StepWithResult<File<FT>> {

public:
  /// Constructor.
  /// \param deleteOnExit If true, delete the temporary file as soon as this
  /// instance is destructed.
  explicit CreateTmpFile(bool deleteOnExit)
      : StepBase(fmt::format("create-tmp-file (*{})",
                             FileSystem::getFileExtension<FT>())),
        deleteFile{deleteOnExit}, file{"tmp", false} {}

  ExecutionResult execute() override {
    file = FileSystem::createTempFile<FT>(deleteFile);
    return success();
  }

  File<FT> *result() override { return &file; }

private:
  bool deleteFile;

  bool valid = false;

  File<FT> file;
};

} // namespace spnc

#endif // SPNC_COMPILER_INCLUDE_DRIVER_PIPELINE_BASICSTEPS_H
