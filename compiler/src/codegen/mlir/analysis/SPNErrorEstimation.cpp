//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPNErrorEstimation.h"

using namespace spnc;
using namespace mlir::spn;

SPNErrorEstimation::SPNErrorEstimation(Operation* root, ERRORMODEL err_model, bool err_relative, double err_margin) :
    rootNode(root), error_model(err_model), relative_error(err_relative), error_margin(err_margin) {
  assert(root);
  analyzeGraph(root);
}

void SPNErrorEstimation::analyzeGraph(Operation* graphRoot) {
  assert(graphRoot);
  iterationCount = 0;
  while (!satisfiedRequirements && !abortAnalysis) {
    format_bits_significance = (error_model == ERRORMODEL::EM_FIXED_POINT) ?
        (2 + iterationCount) : std::get<0>(Float_Formats[iterationCount]);
    EPS = std::pow(BASE_TWO, double(-(format_bits_significance + 1.0)));
    ERR_COEFFICIENT = 1.0 + EPS;

    traverseSubgraph(graphRoot);
    estimateLeastMagnitudeBits();
    satisfiedRequirements = checkRequirements();

    ++iterationCount;
  }
  selectOptimalType();
}

void SPNErrorEstimation::estimateLeastMagnitudeBits() {
  // ToDo: This is executed once. Better way to realize?
  if (!estimatedLeastMagnitudeBits) {
    // Find global extreme values
    for (auto e : spn_node_values) {
      spn_node_value_global_maximum = std::max(spn_node_value_global_maximum, std::get<2>(e.second));
      spn_node_value_global_minimum = std::min(spn_node_value_global_minimum, std::get<3>(e.second));
    }

    switch (error_model) {
      case ERRORMODEL::EM_FLOATING_POINT:
        // Overflow and underflow has to be taken into account -- Overflow is handled by fixed-pt: therefore, no break!
        format_bits_magnitude = (int) std::ceil(std::log2(std::abs(std::log2(spn_node_value_global_minimum))));
      case ERRORMODEL::EM_FIXED_POINT:
        // Only overflow / maximum-value has to be taken into account
        format_bits_magnitude =
            std::max(format_bits_magnitude,
                     (int) std::ceil(std::log2(std::abs(std::log2(spn_node_value_global_maximum)))));
    }

    estimatedLeastMagnitudeBits = true;
  }
}

bool SPNErrorEstimation::checkRequirements() {
  bool satisfied = true;
  auto rootValues = spn_node_values[rootNode];
  double value = std::get<0>(rootValues);
  double defect = std::get<1>(rootValues);

  // Floating-point: Check if currently selected format has enough magnitude bits
  if (error_model == ERRORMODEL::EM_FLOATING_POINT) {
    // If least amount of magnitude bits is greater than currently selected format's: Search for fitting format.
    if (format_bits_magnitude > std::get<1>(Float_Formats[iterationCount])) {
      for (int i = iterationCount + 1; i < NUM_FLOAT_FORMATS; ++i) {
        if (format_bits_magnitude > std::get<1>(Float_Formats[i])) {
          // Each time we check a NON-fitting format we can skip the corresponding iteration.
          ++iterationCount;
        } else {
          // A fitting format was found, next iteration will start with a format that provides enough magnitude bits.
          break;
        }
      }

      // Requirements NOT met.
      satisfied = false;
    }
  }

  if (relative_error) {
    satisfied &= (error_margin > std::abs((value / defect) - 1.0));
  } else {
    satisfied &= (error_margin > std::abs(value - defect));
  }

  // No further floating point format available -- abort.
  if (!satisfied && (error_model == ERRORMODEL::EM_FLOATING_POINT) && (iterationCount >= (NUM_FLOAT_FORMATS - 1))) {
    abortAnalysis = true;
    selectedType = std::get<2>(Float_Formats[NUM_FLOAT_FORMATS - 1]);
    SPDLOG_WARN("Selected floating point format does not meet error requirements, but no further format is available.");
  }

  return satisfied;
}

void SPNErrorEstimation::traverseSubgraph(Operation* subgraphRoot) {
  assert(subgraphRoot);
  auto operands = subgraphRoot->getOperands();

  // Operations with more than one operand -- possible: inner node, e.g. sum or product.
  // Operations with one operand -- possible: leaf node.
  // Operations with no operand -- possible: constant.
  if (operands.size() > 1) {
    arg_t passed_arg(nullptr);

    // First: Visit every child-node.
    for (auto child : operands) {
      traverseSubgraph(child.getDefiningOp());
    }

    // Second: Estimate errors of current operation.
    if (auto sum = dyn_cast<SumOp>(subgraphRoot)) {
      estimateErrorSum(sum);
    } else if (auto product = dyn_cast<ProductOp>(subgraphRoot)) {
      estimateErrorProduct(product);
    }

  } else if (operands.size() == 1) {
    // NOTE: ATM there is only one leaf type, others will have to be added with a dyn_cast<> as well.
    if (auto histogram = dyn_cast<HistogramOp>(subgraphRoot)) {
      estimateErrorHistogram(histogram);
    }
  } else {
    if (auto constant = dyn_cast<ConstantOp>(subgraphRoot)) {
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
    // max = std::max(max, std::get<2>(b.cast<bucket_t>()));
    // min = std::min(min, std::get<2>(b.cast<bucket_t>()));
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

void SPNErrorEstimation::selectOptimalType() {
  if (!abortAnalysis) {
    switch (error_model) {
      case ERRORMODEL::EM_FIXED_POINT:
        selectedType = IntegerType::get((format_bits_magnitude + format_bits_significance), rootNode->getContext());
        break;
      case ERRORMODEL::EM_FLOATING_POINT:
        selectedType = std::get<2>(Float_Formats[iterationCount - 1]);
        break;
    }
  }
}

mlir::Type SPNErrorEstimation::getOptimalType() {
  return selectedType;
}
