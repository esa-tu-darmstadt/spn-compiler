//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_FILESYSTEM_H
#define SPNC_FILESYSTEM_H

#include <functional>
#include <iostream>
#include <string>
#include <stdlib.h>

namespace spnc {

  enum class FileType;
  template<FileType Type>
  class File;

  ///
  /// Helper class providing methods to manage external files.
  class FileSystem {

  public:

    template<FileType Type>
    static File<Type> createTempFile(bool deleteTmpOnExit = true);

    /// Delete the file given by the path.
    /// \param fileName File path of the file to delete.
    static void deleteFile(const std::string& fileName) {
      std::remove(fileName.c_str());
    }

  private:

    explicit FileSystem() = default;

  };

  ///
  /// Enumeration of special file-types.
  enum class FileType { SPN_JSON, LLVM_BC, OBJECT, SHARED_OBJECT, DOT, STAT_JSON, SPN_BINARY };

  using LLVMBitcode = File<FileType::LLVM_BC>;
  using ObjectFile = File<FileType::OBJECT>;
  using SharedObject = File<FileType::SHARED_OBJECT>;
  using StatsFile = File<FileType::STAT_JSON>;
  using BinarySPN = File<FileType::SPN_BINARY>;

  /// File on the file-system.
  /// \tparam Type Type of the file.
  template<FileType Type>
  class File {
  public:
    /// Constructor.
    /// \param fileName Full path to the file.
    /// \param _deleteOnExit Flag to indicate whether this file should be deleted on exit from the compiler.
    /// Defaults to false.
    explicit File(const std::string& fileName, bool _deleteOnExit = false) : fName{fileName},
                                                                             deleteOnExit{_deleteOnExit} {}

    ~File() {
      if (deleteOnExit) {
        FileSystem::deleteFile(fName);
      }
    }

    File(File const&) = delete;

    void operator=(File const&) = delete;

    /// Move constructor.
    /// \param other Move source.
    File(File&& other) noexcept : fName{other.fName}, deleteOnExit{other.deleteOnExit} {
      other.fName = "";
      other.deleteOnExit = false;
    }

    /// Move assignment.
    /// \param other Move source.
    /// \return Reference to the move target.
    File& operator=(File&& other) noexcept {
      fName = std::move(other.fName);
      deleteOnExit = std::move(other.deleteOnExit);
      other.deleteOnExit = false;
    }

    /// Get the path of this file.
    /// \return Path of this file.
    const std::string& fileName() { return fName; }

  private:
    std::string fName;

    bool deleteOnExit;
  };

  /// Helper method to create a temporary file.
  /// \tparam Type FileType of the file.
  /// \param deleteTmpOnExit Flag to indicate whether the created file should be deleted on exit from the compiler.
  /// \return Created File.
  template<FileType Type>
  File<Type> FileSystem::createTempFile(bool deleteTmpOnExit) {
    std::string fileExtension;
    switch (Type) {
      case FileType::SPN_JSON:
      case FileType::STAT_JSON: fileExtension = ".json";
        break;
      case FileType::LLVM_BC: fileExtension = ".bc";
        break;
      case FileType::DOT: fileExtension = ".dot";
        break;
      case FileType::OBJECT: fileExtension = ".o";
        break;
      case FileType::SHARED_OBJECT: fileExtension = ".so";
        break;
      default: fileExtension = "";
    }
    std::string tmpName = "/tmp/spncXXXXXX" + fileExtension;
    auto suffixLength = fileExtension.length();
    // Use the mkstemps function from the Posix standard to create a temporary files with file suffix.
    mkstemps(&tmpName[0], suffixLength);
    return File<Type>{tmpName, deleteTmpOnExit};
  }

}

#endif //SPNC_FILESYSTEM_H
