#include "GraphIRNode.h"

//
// Created by ls on 10/7/19.
//
Histogram::Histogram(std::string id, std::shared_ptr<InputVar> indexVar, const std::vector<HistogramBucket>& buckets):
    GraphIRNode{std::move(id)}, _indexVar{std::move(indexVar)} {
    _buckets = std::make_shared<std::vector<HistogramBucket>>(buckets.size());
    std::copy(buckets.begin(), buckets.end(), _buckets->begin());
}

std::shared_ptr<InputVar> Histogram::indexVar() const {return _indexVar;}

std::shared_ptr<std::vector<HistogramBucket>> Histogram::buckets() const {return _buckets;}
