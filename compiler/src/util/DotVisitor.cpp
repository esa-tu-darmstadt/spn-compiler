//
// Created by ls on 10/9/19.
//

#include <fstream>
#include "DotVisitor.h"

namespace spnc {

    DotVisitor::DotVisitor(ActionWithOutput<IRGraph>& _input, const std::string &outputFile)
      : ActionSingleInput<IRGraph, File<FileType::DOT>>{_input}, outfile{outputFile} {}

    void DotVisitor::writeDotGraph(const NodeReference& rootNode) {
      rootNode->accept(*this, nullptr);
      std::ofstream fileStream;
      fileStream.open(outfile.fileName());
      fileStream << "digraph spn {" << std::endl;
      fileStream << nodes.rdbuf();
      fileStream << edges.rdbuf();
      fileStream << "}" << std::endl;
    }

    void DotVisitor::visitInputvar(InputVar &n, arg_t arg) {
      nodes << "v" << n.id() << " [shape=box, label=\"input " << n.id() << " [" << n.index() << "]\"];" << std::endl;
    }

    void DotVisitor::visitHistogram(Histogram &n, arg_t arg) {
      nodes << "v" << n.id() << " [shape=box, label=\"histogram " << n.id();
      nodes << "\\n #buckets: " << n.buckets()->size() << "\"];" << std::endl;
      edges << "v" << n.id() << " -> v" << n.indexVar()->id() << ";" << std::endl;
      n.indexVar()->accept(*this, nullptr);
    }

    void DotVisitor::visitProduct(Product &n, arg_t arg) {
      nodes << "v" << n.id() << " [shape=box, label=\"product " << n.id() << "\"];" << std::endl;
      for(auto& child : *n.multiplicands()){
        edges << "v" << n.id() << " -> v" << child->id() << ";" << std::endl;
        child->accept(*this, nullptr);
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
        edges << "v" << n.id() << " -> v" << child.addend->id() << " [label=\"" << child.weight << "\"];" << std::endl;
        child.addend->accept(*this, nullptr);
      }
    }

    spnc::File<FileType::DOT>& spnc::DotVisitor::execute() {
      if(!cached){
        writeDotGraph(input.execute().rootNode);
      }
      return outfile;
    }
}
