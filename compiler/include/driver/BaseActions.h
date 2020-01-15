//
// Created by ls on 1/14/20.
//

#ifndef SPNC_BASEACTIONS_H
#define SPNC_BASEACTIONS_H

#include <fstream>
#include <memory>
#include "Actions.h"

namespace spnc {

    enum class FileType{SPN_JSON, LLVM_BC, OBJECT, SHARED_OBJECT, DOT};

    template<FileType Type>
    class File {
    public:
        explicit File(const std::string& fileName) : fName{fileName}, fileStream{fileName} {}

        std::ifstream& stream() { return fileStream; }

        const std::string& fileName() { return fName; }

    private:
        std::string fName;

        std::ifstream fileStream;
    };

    template<FileType Type>
    class FileInputAction : public ActionWithOutput<File<Type>> {
    public:
        explicit FileInputAction(const std::string& fileName) : file{fileName} {}

        File<Type>& execute() override {
          return file;
        }

    private:
        File<Type> file;

    };

    class StringInputAction : public ActionWithOutput<std::string> {
    public:
        explicit StringInputAction(const std::string& s) : content{std::make_unique<std::string>(s)}{}

        std::string& execute() override { return *content; }

    private:
        std::unique_ptr<std::string> content;
    };
}




#endif //SPNC_BASEACTIONS_H
