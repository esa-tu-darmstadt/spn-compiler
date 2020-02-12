//
// Created by ls on 10/8/19.
//

#include "GraphIRNode.h"
#include <graph-ir/transform/Visitor.h>

namespace spnc {

    WeightedSum::WeightedSum(std::string id, const std::vector <WeightedAddend> &addends) : GraphIRNode(std::move(id)) {
      _addends = std::make_shared<std::vector<WeightedAddend>>(addends.size());
      std::copy(addends.begin(), addends.end(), _addends->begin());
    }

    std::shared_ptr<std::vector<WeightedAddend>> WeightedSum::addends() const {return _addends;}

    void WeightedSum::accept(Visitor& visitor, arg_t arg) {
      return visitor.visitWeightedSum(*this, arg);
    }
}
