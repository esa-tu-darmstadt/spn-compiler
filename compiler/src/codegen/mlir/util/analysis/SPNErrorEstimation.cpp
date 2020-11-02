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
  ERR_COEFFICIENT = 1.0 + EPS;
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
  double value = 0.0, defect = 0.0, max = 0.0, min = 0.0;

  // Assumption: There are exactly & guaranteed(!) two operands
  auto operands = op.getOperands();
  assert(operands.size() == 2);

  // Assumption: Both operands were encountered beforehand -- i.e. their values are known.
  auto it_v1 = spn_node_values.find(operands[0].getDefiningOp());
  auto it_v2 = spn_node_values.find(operands[1].getDefiningOp());
  assert(it_v1 != spn_node_values.end());
  assert(it_v2 != spn_node_values.end());

  // Retrieve the corresponding tuples
  std::tuple<double,double,double,double,int>& t1 = it_v1->second;
  std::tuple<double,double,double,double,int>& t2 = it_v2->second;

  value = std::get<0>(t1) + std::get<0>(t2);

  int max_depth = std::max(std::get<4>(t1), std::get<4>(t2)) + 1;

  switch (error_model) {
    case ERRORMODEL::EM_FIXED_POINT:
      defect = std::get<1>(t1) + std::get<1>(t2);
      break;
    case ERRORMODEL::EM_FLOATING_POINT:
      defect = value * std::pow(ERR_COEFFICIENT, max_depth);
      break;
  }

  // "max": values will be added
  // "min": the higher value will be ignored entirely (adder becomes min-selection)
  max = std::get<2>(t1) + std::get<2>(t2);
  min = std::min(std::get<3>(t1), std::get<3>(t2));

  spn_node_values.emplace(op.getOperation(), std::make_tuple(value, defect, max, min, max_depth));
}

void SPNErrorEstimation::estimateErrorProduct(ProductOp op) {
  double value = 0.0, defect = 0.0, max = 0.0, min = 0.0;
  // Tuple from "value map": { accurate, defective, max, min }

  // Assumption: There are exactly & guaranteed(!) two operands
  auto operands = op.getOperands();
  assert(operands.size() == 2);

  // Assumption: Both operands were encountered beforehand -- i.e. their values are known.
  auto it_v1 = spn_node_values.find(operands[0].getDefiningOp());
  auto it_v2 = spn_node_values.find(operands[1].getDefiningOp());
  assert(it_v1 != spn_node_values.end());
  assert(it_v2 != spn_node_values.end());

  // Retrieve the corresponding tuples
  std::tuple<double,double,double,double,int>& t1 = it_v1->second;
  std::tuple<double,double,double,double,int>& t2 = it_v2->second;

  value = std::get<0>(t1) * std::get<0>(t2);
  defect = value;

  int max_depth = std::max(std::get<4>(t1), std::get<4>(t2)) + 1;

  // Moved declaration(s) out of switch-case to avoid errors / warnings.
  double delta_t1, delta_t2;
  int accumulated_depth;

  switch (error_model) {
    case ERRORMODEL::EM_FIXED_POINT:
      // Calculate deltas of the given operators (see equation (5) of ProbLP paper)
      delta_t1 = std::get<1>(t1) - std::get<0>(t1);
      delta_t2 = std::get<1>(t2) - std::get<0>(t2);

      // ToDo: Asserts should trivially be satisfied. Remove when confirmed.
      assert(delta_t1 >= 0.0);
      assert(delta_t2 >= 0.0);

      // Add the corresponding total delta to the already calculated ("accurate") value
      defect += (std::get<2>(t1) * delta_t2) + (std::get<2>(t2) * delta_t1) + (delta_t1 * delta_t2) + EPS;
      break;
    case ERRORMODEL::EM_FLOATING_POINT:
      accumulated_depth = std::get<4>(t1) + std::get<4>(t2) + 1;
      defect *= std::pow(ERR_COEFFICIENT, accumulated_depth);
      break;
  }

  // "max": highest values will be multiplied
  // "min": lowest values will be multiplied
  max = std::get<2>(t1) * std::get<2>(t2);
  min = std::get<3>(t1) * std::get<3>(t2);

  spn_node_values.emplace(op.getOperation(), std::make_tuple(value, defect, max, min, max_depth));
}

void SPNErrorEstimation::estimateErrorHistogram(HistogramOp op) {
  auto buckets = op.buckets();

  double value;
  double max = std::numeric_limits<double>::min();
  double min = std::numeric_limits<double>::max();
  double defect;

  for (auto b : buckets) {
    max = std::max(max, std::get<2>(b.cast<bucket_t>()));
    min = std::min(min, std::get<2>(b.cast<bucket_t>()));
  }

  // Select maximum as "value"
  value = max;

  switch (error_model) {
    case ERRORMODEL::EM_FIXED_POINT:
      defect = value + EPS;
      break;
    case ERRORMODEL::EM_FLOATING_POINT:
      defect = value * ERR_COEFFICIENT;
      break;
  }

  // ToDo: Check if helpful / needed -- Check for changed values.
  // ToDo: This could also signal that no buckets were encountered.
  assert(max != std::numeric_limits<double>::min());
  assert(min != std::numeric_limits<double>::max());

  spn_node_values.emplace(op.getOperation(), std::make_tuple(value, defect, max, min, 1));
}

void SPNErrorEstimation::estimateErrorConstant(ConstantOp op) {
  double value = op.value().convertToDouble();
  double max = value;
  double min = value;
  double defect = 0.0;

  switch (error_model) {
    case ERRORMODEL::EM_FIXED_POINT:
      defect = value + EPS;
      break;
    case ERRORMODEL::EM_FLOATING_POINT:
      defect = value * ERR_COEFFICIENT;
      break;
  }

  spn_node_values.emplace(op.getOperation(), std::make_tuple(value, defect, max, min, 1));
}
