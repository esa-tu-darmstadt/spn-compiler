//
// Created by ls on 10/7/19.
//

#include <graph-ir/GraphIRNode.h>
#include "../transform/Visitor.h"

namespace spnc {

  Histogram::Histogram(std::string id, InputVar* indexVar, const std::vector <HistogramBucket>& bs) :
      GraphIRNode{std::move(id)}, _indexVar{indexVar} {
    std::copy(bs.begin(), bs.end(), std::back_inserter(_buckets));
  }

  InputVar& Histogram::indexVar() const { return *_indexVar; }

  const std::vector <HistogramBucket>& Histogram::buckets() const { return _buckets; }

  void Histogram::accept(Visitor& visitor, arg_t arg) {
    return visitor.visitHistogram(*this, arg);
  }
}
