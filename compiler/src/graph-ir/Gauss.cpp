//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <graph-ir/GraphIRNode.h>
#include <graph-ir/transform/Visitor.h>

using namespace spnc;

Gauss::Gauss(std::string id, InputVar* indexVar, double mean,
          double stddev)
    : GraphIRNode{std::move(id)}, _indexVar{indexVar}, _mean(mean),
      _stddev(stddev) {}

InputVar& Gauss::indexVar() const {return *_indexVar;}

double Gauss::mean() const {return _mean;}
double Gauss::stddev() const {return _stddev;}

void Gauss::accept(Visitor& visitor, arg_t arg) {
    return visitor.visitGauss(*this, arg);
}
