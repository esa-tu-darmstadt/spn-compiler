//
// Created by ls on 1/16/20.
//

#ifndef SPNC_FILESYSTEM_H
#define SPNC_FILESYSTEM_H

#include <functional>
#include <iostream>
#include <string>

namespace spnc {


    enum class FileType;
    template<FileType Type>
    class File;

    class FileSystem {

    public:

        template<FileType Type>
        static File<Type> createTempFile(bool deleteTmpOnExit = true);

        static void deleteFile(const std::string& fileName){
          std::remove(fileName.c_str());
        }

    private:

        explicit FileSystem() = default;

    };

    enum class FileType{SPN_JSON, LLVM_BC, OBJECT, SHARED_OBJECT, DOT};

    template<FileType Type>
    class File {
    public:
        explicit File(const std::string& fileName, bool _deleteOnExit = false) : fName{fileName},
                                                                                 deleteOnExit{_deleteOnExit} {}

        ~File(){
          if(deleteOnExit){
            FileSystem::deleteFile(fName);
          }
        }

        File(File const&) = delete;

        void operator=(File const&) = delete;

        File(File&& other) noexcept : fName{other.fName}, deleteOnExit{other.deleteOnExit} {
          other.fName = "";
          other.deleteOnExit = false;
        }

        const std::string& fileName() { return fName; }

    private:
        std::string fName;

        bool deleteOnExit;
    };

    template<FileType Type>
    File<Type> FileSystem::createTempFile(bool deleteTmpOnExit) {
      std::string fileExtension;
      switch(Type){
        case FileType::SPN_JSON:      fileExtension = ".json"; break;
        case FileType::LLVM_BC:       fileExtension = ".bc"; break;
        case FileType::DOT:           fileExtension = ".dot"; break;
        case FileType::OBJECT:        fileExtension = ".o"; break;
        case FileType::SHARED_OBJECT: fileExtension = ".so"; break;
        default:                      fileExtension = "";
      }
      std::string tmpName = std::string{std::tmpnam(nullptr)} + fileExtension;
      return File<Type>{tmpName, deleteTmpOnExit};
    }

}



#endif //SPNC_FILESYSTEM_H
