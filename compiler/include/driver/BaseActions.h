//
// Created by ls on 1/14/20.
//

#ifndef SPNC_BASEACTIONS_H
#define SPNC_BASEACTIONS_H

#include <fstream>
#include <memory>
#include "Actions.h"
#include <util/FileSystem.h>

namespace spnc {

    template<FileType Type>
    class FileInputAction : public ActionWithOutput<std::string> {
    public:
        explicit FileInputAction(const std::string& _fileName) : fileName{_fileName} {}

        std::string& execute() override {
          if(!cached){
            content = std::make_unique<std::string>();
            std::ifstream in(fileName);
            if(!in){
              std::cerr << "ERROR: Could not read file " << fileName << std::endl;
              throw std::system_error{};
            }
            in.seekg(0, std::ios::end);
            content->resize(in.tellg());
            in.seekg(0, std::ios::beg);
            in.read(&(*content)[0], content->size());
            in.close();
          }
          return *content;
        }

    private:
        std::string fileName;
        std::unique_ptr<std::string> content;
        bool cached = false;

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
