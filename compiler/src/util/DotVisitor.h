//
// Created by ls on 10/9/19.
//

#ifndef SPNC_DOTVISITOR_H
#define SPNC_DOTVISITOR_H


#include <transform/BaseVisitor.h>
#include <sstream>
#include <driver/Actions.h>
#include <driver/BaseActions.h>

namespace spnc {

    class DotVisitor : BaseVisitor, ActionSingleInput<IRGraph, File<FileType::DOT>> {

    public:

        explicit DotVisitor(ActionWithOutput<IRGraph>& _input, const std::string& outputFile);

        File<FileType::DOT> &execute() override;

    private:

        void writeDotGraph(const NodeReference& rootNode);

    public:

        void visitInputvar(InputVar& n, arg_t arg) override ;

        void visitHistogram(Histogram& n, arg_t arg) override ;

        void visitProduct(Product& n, arg_t arg) override ;

        void visitSum(Sum& n, arg_t arg) override ;

        void visitWeightedSum(WeightedSum& n, arg_t arg) override ;

    private:
        std::stringstream nodes{};
        std::stringstream edges{};
        const std::string& outfile;
        bool cached = false;
        File<FileType::DOT> file = File<FileType::DOT>("uninitialized.dot");
    };
}




#endif //SPNC_DOTVISITOR_H
