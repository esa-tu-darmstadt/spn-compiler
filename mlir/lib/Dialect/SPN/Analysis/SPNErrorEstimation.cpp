//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <iostream>
#include "SPN/Analysis/SPNErrorEstimation.h"

using namespace mlir::spn;

SPNErrorEstimation::SPNErrorEstimation(Operation* root, ERRORMODEL err_model, bool err_relative, double err_margin) :
    rootNode(root), error_model(err_model), relative_error(err_relative), error_margin(err_margin) {
  std::cout << "SPNErrorEstimation ctor entry" << std::endl;
  assert(root);
  iterationCount = 0;
  while (!satisfiedRequirements && !abortAnalysis) {
    analyzeGraph(root);
    ++iterationCount;
  }
  selectOptimalType();
  std::cout << "SPNErrorEstimation ctor exit" << std::endl;
}

void SPNErrorEstimation::analyzeGraph(Operation* graphRoot) {
  assert(graphRoot);
  // Either select next floating-point format or increase number of Integer-bits
  format_bits_significance = (error_model == ERRORMODEL::EM_FLOATING_POINT) ?
      std::get<0>(Float_Formats[iterationCount]) : (2 + iterationCount);
  EPS = std::pow(BASE_TWO, double(-(format_bits_significance + 1.0)));
  ERR_COEFFICIENT = 1.0 + EPS;

  traverseSubgraph(graphRoot);

  if (!estimatedLeastMagnitudeBits) {
    estimateLeastMagnitudeBits();
  }

  satisfiedRequirements = checkRequirements();
}

void SPNErrorEstimation::estimateLeastMagnitudeBits() {
  // Find global extreme values
  for (auto entry : spn_node_values) {
    spn_node_value_global_maximum = std::max(spn_node_value_global_maximum, std::get<2>(entry.second));
    spn_node_value_global_minimum = std::min(spn_node_value_global_minimum, std::get<3>(entry.second));
  }

  switch (error_model) {
    case ERRORMODEL::EM_FLOATING_POINT:
      // Overflow and underflow has to be taken into account.
      // ToDo: CONFIRM -- Since 2's complement is used, an increment of 1 is necessary to estimate the correct number.
      format_bits_magnitude = (int) std::ceil(1 + std::log2(std::abs(std::log2(spn_node_value_global_minimum))));
      format_bits_magnitude =
          std::max(format_bits_magnitude,
                   (int) std::ceil(1 + std::log2(std::abs(std::log2(spn_node_value_global_maximum)))));
      break;
    case ERRORMODEL::EM_FIXED_POINT:
      // Only overflow / maximum-value has to be taken into account
      format_bits_magnitude = (int) std::ceil(std::log2(spn_node_value_global_maximum));
      break;
  }

  std::cerr << "--------------------------------------------------" << std::endl;
  std::cerr << "(" << iterationCount << ".E) format_bits_magnitude: " << format_bits_magnitude << std::endl;
  std::cerr << "(" << iterationCount << ".E) spn_node_value_global_minimumX: " << spn_node_value_global_minimum << std::endl;
  std::cerr << "(" << iterationCount << ".E) spn_node_value_global_maximumX: " << spn_node_value_global_maximum << std::endl;
  std::cerr << "==================================================" << std::endl;

  estimatedLeastMagnitudeBits = true;
}

bool SPNErrorEstimation::checkRequirements() {
  bool satisfied = true;
  auto rootValues = spn_node_values[rootNode];
  double value = std::get<0>(rootValues);
  double defect = std::get<1>(rootValues);

  /*
  std::cerr << "--------------------------------------------------" << std::endl;
  std::cerr << "(" << iterationCount << ".4) format_bits_magnitude: " << format_bits_magnitude << std::endl;
  std::cerr << "(" << iterationCount << ".4) spn_node_value_global_minimum: " << spn_node_value_global_minimum << std::endl;
  std::cerr << "(" << iterationCount << ".4) spn_node_value_global_maximum: " << spn_node_value_global_maximum << std::endl;
  std::cerr << "==================================================" << std::endl;
   */

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

  /*
  std::cerr << "--------------------------------------------------" << std::endl;
  std::cerr << "(" << iterationCount << ") satisfied:             " << satisfied << std::endl;
  std::cerr << "(" << iterationCount << ") format_bits_magnitude: " << format_bits_magnitude << std::endl;
  std::cerr << "(" << iterationCount << ") relative_error?        " << relative_error << std::endl;
  std::cerr << "(" << iterationCount << ") ERR-margin:            " << error_margin << std::endl;
  std::cerr << "(" << iterationCount << ") ERR-rel:               " << std::abs((value / defect) - 1.0) << std::endl;
  std::cerr << "(" << iterationCount << ") ERR-abs:               " << std::abs(value - defect) << std::endl;
  std::cerr << "==================================================" << std::endl;
   */

  // No further floating point format available -- abort.
  if (!satisfied && (error_model == ERRORMODEL::EM_FLOATING_POINT) && (iterationCount >= (NUM_FLOAT_FORMATS - 1))) {
    abortAnalysis = true;
    selectedType = std::get<2>(Float_Formats[NUM_FLOAT_FORMATS - 1])(rootNode->getContext());
    // SPDLOG_WARN("Selected floating point format does not meet error requirements, but no further format is available.");
  }

  return satisfied;
}

void SPNErrorEstimation::traverseSubgraph(Operation* subgraphRoot) {
  assert(subgraphRoot);
  auto operands = subgraphRoot->getOperands();

  std::cerr << "traverseSubgraph of ";
  subgraphRoot->dump();
  std::cerr << std::endl;

  if (auto query = dyn_cast<mlir::spn::QueryInterface>(subgraphRoot)) {
    std::cerr << "QueryInterface!" << std::endl;
  }

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
  std::cout << "estimateErrorSum entry" << std::endl;
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
  std::cout << "estimateErrorSum exit" << std::endl;
}

void SPNErrorEstimation::estimateErrorProduct(ProductOp op) {
  std::cout << "estimateErrorProduct entry" << std::endl;
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
  std::cout << "estimateErrorProduct exit" << std::endl;
}

void SPNErrorEstimation::estimateErrorHistogram(HistogramOp op) {
  std::cout << "estimateErrorHistogram entry" << std::endl;
  auto buckets = op.buckets().getValue();

  double value;
  double max = std::numeric_limits<double>::min();
  double min = std::numeric_limits<double>::max();
  double defect;

  op.dump();

  for (auto b : op.buckets().getValue()) {
    b.dump();

    /*
    auto bucket = dyn_cast<bucket_t>(b);
    // auto bucket = dyn_cast<bucket_t>(b);
    max = std::max(max, std::get<2>(bucket));
    min = std::min(min, std::get<2>(bucket));
    */
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
  std::cout << "estimateErrorHistogram exit" << std::endl;
}

void SPNErrorEstimation::estimateErrorConstant(ConstantOp op) {
  std::cout << "estimateErrorConstant entry" << std::endl;
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
  std::cout << "estimateErrorConstant exit" << std::endl;
}

void SPNErrorEstimation::selectOptimalType() {
  std::cout << "abortAnalysis: " << abortAnalysis << std::endl;
  std::cout << "iterationCount: " << iterationCount << std::endl;
  std::cout << "format_bits_magnitude: " << format_bits_magnitude << std::endl;
  std::cout << "format_bits_significance: " << format_bits_significance << std::endl;

  if (!abortAnalysis) {
    switch (error_model) {
      case ERRORMODEL::EM_FIXED_POINT:
        selectedType =
            IntegerType::get((format_bits_magnitude + format_bits_significance), rootNode->getContext());
        break;
      case ERRORMODEL::EM_FLOATING_POINT:
        // Invoke call: Type::get(MLIRContext*)
        selectedType = std::get<2>(Float_Formats[iterationCount - 1])(rootNode->getContext());
        break;
    }
  }
}

mlir::Type SPNErrorEstimation::getOptimalType() {
  return selectedType;
}
