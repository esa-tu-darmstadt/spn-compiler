//
// Created by ls on 10/8/19.
//

#include <graph-ir/transform/Visitor.h>
#include <graph-ir/GraphIRNode.h>

namespace spnc {

  Product::Product(std::string id, const std::vector<NodeReference>& multiplicands) : GraphIRNode{std::move(id)} {
    std::copy(multiplicands.begin(), multiplicands.end(), std::back_inserter(_multiplicands));
  }

  const std::vector<NodeReference>& Product::multiplicands() const { return _multiplicands; }

    void Product::accept(Visitor& visitor, arg_t arg) {
      return visitor.visitProduct(*this, arg);
    }
}
