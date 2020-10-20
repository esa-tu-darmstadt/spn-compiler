//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPNErrorEstimation.h"

using namespace mlir;
using namespace mlir::spn;
// using namespace spnc;

SPNErrorEstimation::SPNErrorEstimation() : root(nullptr), error_margin(0.0), relative_error(false) {}

SPNErrorEstimation::SPNErrorEstimation(Operation* _root, double _error_margin, bool _relative_error, ERRORMODEL _em) :
  root(_root), error_margin(std::abs(_error_margin)), relative_error(_relative_error), error_model(_em) {
  // ToDo: Move and set correct number of bits for each calculation.
  format_bits_significance = 10;
  format_bits_magnitude = 5;
  // ToDo: Move calculation.
  EPS = std::pow(BASE_TWO, double(-(format_bits_significance + 1.0)));
  update();
}

void SPNErrorEstimation::update() {
  for (auto e : errors) {
    e = { 0.0, 0.0 };
  }

  std::shared_ptr<void> passed_arg(nullptr);
  visitNode(root, passed_arg);

  processResults();
}

void SPNErrorEstimation::visitNode(Operation* op, const arg_t& arg) {
  if (op == nullptr) {
    // Encountered nullptr -- abort.
    return;
  }

  // auto information = std::static_pointer_cast<SomeCrucialInfo>(arg)->info;

  auto operands = op->getOperands();

  // Operations with more than one operand -- possible: inner node, e.g. sum or product.
  // Operations with one operand -- possible: leaf node.
  // Operations with no operand -- possible: constant.
  if (operands.size() > 1) {
    arg_t passed_arg(nullptr);

    // First: Visit every child-node.
    for (auto child : operands) {
      visitNode(child.getDefiningOp(), passed_arg);
    }

    // Second: Estimate errors of current operation.
    if (auto sum = dyn_cast<SumOp>(op)) {
      estimateErrorSum(sum);
    } else if (auto product = dyn_cast<ProductOp>(op)) {
      estimateErrorProduct(product);
    }

  } else if (operands.size() == 1) {
    // NOTE: ATM there is only one leaf type, others will have to be added with a dyn_cast<> as well.
    if (auto histogram = dyn_cast<HistogramOp>(op)) {
      estimateErrorHistogram(histogram);
    }
  } else {
    if (auto constant = dyn_cast<ConstantOp>(op)) {
      estimateErrorConstant(constant);
    }
  }
}

void SPNErrorEstimation::estimateErrorSum(SumOp op) {

  double value = 0.0;
  double max = 0.0;
  double min = std::numeric_limits<double>::max();
  double error = (error_model == ERRORMODEL::EM_FIXED_POINT) ? 0.0 : EPS;
  std::tuple<double,double,double,double> t;

  for (auto o : op.getOperands()) {
    auto child = o.getDefiningOp();
    auto it = spn_node_values.find(child);

    if (it != spn_node_values.end()) {
      t = it->second;
    } else {
      // SPDLOG_WARN();
      // return;
    }

    value += std::get<0>(t);

    switch (error_model) {
      case ERRORMODEL::EM_FIXED_POINT:
        error += std::get<1>(t);
        break;
      case ERRORMODEL::EM_FLOATING_POINT:
        // ToDo: Error propagation.
        break;
    }

    max += std::get<2>(t);
    min = std::min(min, std::get<3>(t));
  }

}

void SPNErrorEstimation::estimateErrorProduct(ProductOp op) {

  switch (error_model) {
    case ERRORMODEL::EM_FIXED_POINT:
    break;
    case ERRORMODEL::EM_FLOATING_POINT:
    break;
  }

}

void SPNErrorEstimation::estimateErrorHistogram(HistogramOp op) {
  auto buckets = op.buckets();
  auto it_mid = buckets.begin();
  auto it_end = buckets.end();

  // ToDo: Define correct value selection.
  if (buckets.size() > 2) {
    it_mid += (int) (buckets.size() - 1) / 2;
  }

  double value = std::get<2>(it_mid->cast<bucket_t>());
  // Need to decrement the iterator: end() points behind the actual element
  double max = std::get<2>((--it_end)->cast<bucket_t>());
  double min = std::get<2>(op.buckets().begin()->cast<bucket_t>());
  double error = EPS;

  if (error_model == ERRORMODEL::EM_FLOATING_POINT) {
    error *= value;
  }

  spn_node_values.emplace(op.getOperation(), std::make_tuple(value, error, max, min));
}

void SPNErrorEstimation::estimateErrorConstant(ConstantOp op) {
  double value = op.value().convertToDouble();
  double max = value;
  double min = value;
  double error = EPS;

  if (error_model == ERRORMODEL::EM_FLOATING_POINT) {
    error *= value;
  }

  spn_node_values.emplace(op.getOperation(), std::make_tuple(value, error, max, min));
}
