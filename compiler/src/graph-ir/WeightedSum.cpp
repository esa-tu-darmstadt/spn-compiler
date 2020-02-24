//
// Created by ls on 10/8/19.
//

#include <graph-ir/GraphIRNode.h>
#include "../transform/Visitor.h"

namespace spnc {

    WeightedSum::WeightedSum(std::string id, const std::vector <WeightedAddend> &addends) : GraphIRNode(std::move(id)) {
      std::copy(addends.begin(), addends.end(), std::back_inserter(_addends));
    }

  const std::vector<WeightedAddend>& WeightedSum::addends() const { return _addends; }

    void WeightedSum::accept(Visitor& visitor, arg_t arg) {
      return visitor.visitWeightedSum(*this, arg);
    }
}
