#include "GraphIRNode.h"
#include "../transform/Visitor.h"

//
// Created by ls on 10/7/19.
//
Gauss::Gauss(std::string id, std::shared_ptr<InputVar> indexVar, double mean,
             double stddev)
    : GraphIRNode{std::move(id)}, _indexVar{std::move(indexVar)}, _mean(mean),
      _stddev(stddev) {}

std::shared_ptr<InputVar> Gauss::indexVar() const {return _indexVar;}

double Gauss::mean() const {return _mean;}
double Gauss::stddev() const {return _stddev;}

void Gauss::accept(Visitor& visitor, arg_t arg) {
    return visitor.visitGauss(*this, arg);
}
