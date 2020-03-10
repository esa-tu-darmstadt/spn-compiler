//
// Created by ls on 10/8/19.
//

#include <graph-ir/transform/Visitor.h>
#include <graph-ir/GraphIRNode.h>

namespace spnc {

  Sum::Sum(std::string id, const std::vector<NodeReference>& addends) : GraphIRNode{std::move(id)} {
    std::copy(addends.begin(), addends.end(), std::back_inserter(_addends));
  }

  const std::vector<NodeReference>& Sum::addends() const { return _addends; }

    void Sum::accept(Visitor& visitor, arg_t arg) {
      return visitor.visitSum(*this, arg);
    }
}
