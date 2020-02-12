#include "GraphIRNode.h"
#include <graph-ir/transform/Visitor.h>

//
// Created by ls on 10/7/19.
//
namespace spnc {

    Histogram::Histogram(std::string id, std::shared_ptr<InputVar> indexVar, const std::vector<HistogramBucket>& buckets):
            GraphIRNode{std::move(id)}, _indexVar{std::move(indexVar)} {
      _buckets = std::make_shared<std::vector<HistogramBucket>>(buckets.size());
      std::copy(buckets.begin(), buckets.end(), _buckets->begin());
    }

    std::shared_ptr<InputVar> Histogram::indexVar() const {return _indexVar;}

    std::shared_ptr<std::vector<HistogramBucket>> Histogram::buckets() const {return _buckets;}

    void Histogram::accept(Visitor& visitor, arg_t arg) {
      return visitor.visitHistogram(*this, arg);
    }
}
