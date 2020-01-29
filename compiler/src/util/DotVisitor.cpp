//
// Created by ls on 10/9/19.
//

#include <fstream>
#include "DotVisitor.h"

#define NO_HISTOS false

void DotVisitor::writeDotGraph(const NodeReference& rootNode, const std::string& outputFile) {
    rootNode->accept(*this, nullptr);
    std::ofstream fileStream;
    fileStream.open(outputFile);
    fileStream << "digraph spn {" << std::endl;
    fileStream << nodes.rdbuf();
    fileStream << edges.rdbuf();
    fileStream << "}" << std::endl;
}

      struct isHistoVisitor : public BaseVisitor {
        void visitHistogram(Histogram &n, arg_t arg) { isHisto = true; }

        void visitProduct(Product &n, arg_t arg) { isHisto = false; }
        void visitSum(Sum &n, arg_t arg) { isHisto = false; }
        void visitWeightedSum(WeightedSum &n, arg_t arg) { isHisto = false; }
        bool isHisto = false;
      };

void DotVisitor::visitInputvar(InputVar &n, arg_t arg) {
    nodes << "v" << n.id() << " [shape=box, label=\"input " << n.id() << " [" << n.index() << "]\"];" << std::endl;
}

void DotVisitor::visitHistogram(Histogram &n, arg_t arg) {
    nodes << "v" << n.id() << " [shape=box, label=\"histogram " << n.id();
    nodes << "\\n #buckets: " << n.buckets()->size() << " \\n #input:" << n.indexVar()->id() << "  \"];" << std::endl;
    //edges << "v" << n.id() << " -> v" << n.indexVar()->id() << ";" << std::endl;
    //n.indexVar()->accept(*this, nullptr);
}

void DotVisitor::visitProduct(Product &n, arg_t arg) {
    nodes << "v" << n.id() << " [shape=box, label=\"product " << n.id() << "\"];" << std::endl;
    for(auto& child : *n.multiplicands()){

      isHistoVisitor histoCheck;
      child->accept(histoCheck, {});
      if (!(NO_HISTOS) || !histoCheck.isHisto) {
        edges << "v" << n.id() << " -> v" << child->id() << ";" << std::endl;
        child->accept(*this, nullptr);
      }
    }
}

void DotVisitor::visitSum(Sum &n, arg_t arg) {
    nodes << "v" << n.id() << " [shape=box, label=\"sum " << n.id() << "\"];" << std::endl;
    for(auto& child : *n.addends()){
        edges << "v" << n.id() << " -> v" << child->id() << ";" << std::endl;
        child->accept(*this, nullptr);
    }
}

void DotVisitor::visitWeightedSum(WeightedSum &n, arg_t arg) {
    nodes << "v" << n.id() << " [shape=box, label=\"sum " << n.id() << "\"];" << std::endl;
    for(auto& child : *n.addends()){
      isHistoVisitor histoCheck;
      child.addend->accept(histoCheck, {});
      if (!histoCheck.isHisto) {
        edges << "v" << n.id() << " -> v" << child.addend->id() << " [label=\"" << child.weight << "\"];" << std::endl;
        child.addend->accept(*this, nullptr);
      }
    }
}

