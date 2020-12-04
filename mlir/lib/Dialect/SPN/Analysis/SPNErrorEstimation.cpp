//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <iostream>
#include <SPN/SPNAttributes.h>
#include "SPN/Analysis/SPNErrorEstimation.h"

using namespace mlir::spn;

SPNErrorEstimation::SPNErrorEstimation(Operation* root) {
  // std::cerr << "SPNErrorEstimation ctor entry" << std::endl;
  assert(root);

  if (auto module = dyn_cast<mlir::ModuleOp>(root)) {
    module.walk([this](Operation* op){
      if (auto query = dyn_cast<QueryInterface>(op)) {
        queries.push_back(op);
        for (auto r : query.getRootNodes()) {
          roots.push_back(r->getOperand(0).getDefiningOp());
        }
      }
    });
  } else {
    std::cerr << "SPNErrorEstimation ctor expected: mlir::ModuleOp" << std::endl;
    return;
  }

  if (queries.empty()) {
    std::cerr << "Did not find any queries to analyze!" << std::endl;
    return;
  }

  if (queries.size() > 1) {
    std::cerr << "Found more than one query, will only analyze the first query!" << std::endl;
  }

  auto query = dyn_cast<QueryInterface>(queries.front());

  rootNode = roots.front();
  error_model = ERRORMODEL::EM_FLOATING_POINT;
  relative_error = query.getErrorModel();
  error_margin = query.getMaxError();

  // ToDo: Set iterationCount to zero.
  iterationCount = 0;
  while (!satisfiedRequirements && !abortAnalysis) {
    analyzeGraph(rootNode);
    ++iterationCount;
  }
  selectOptimalType();

  auto rootValues = spn_node_values[rootNode];
  double value = std::get<0>(rootValues);
  double defect = std::get<1>(rootValues);

  double err_rel = std::abs((value / defect) - 1.0);
  double err_abs = std::abs(value - defect);
  double err_log_abs = std::log((1 + std::abs((value / defect) - 1.0)));

  std::cerr << "===================== SPNErrorEstimation =====================" << std::endl;
  std::cerr << "|| abortAnalysis:            " << abortAnalysis << std::endl;
  std::cerr << "|| satisfiedRequirements:    " << satisfiedRequirements << std::endl;
  std::cerr << "|| iterationCount:           " << iterationCount << std::endl;
  std::cerr << "|| format_bits_magnitude:    " << format_bits_magnitude << std::endl;
  std::cerr << "|| format_bits_significance: " << format_bits_significance << std::endl;
  std::cerr << "|| EPS:                      " << EPS << std::endl;
  std::cerr << "|| ERR_COEFFICIENT:          " << ERR_COEFFICIENT << std::endl;
  std::cerr << "|| err_margin:               " << error_margin << std::endl;
  std::cerr << "|| spn_node_global_maximum:  " << spn_node_value_global_maximum << std::endl;
  std::cerr << "|| spn_node_global_minimum:  " << spn_node_value_global_minimum << std::endl;
  std::cerr << "|| spn_root_value:           " << std::get<0>(rootValues) << std::endl;
  std::cerr << "|| spn_root_defect:          " << std::get<1>(rootValues) << std::endl;
  std::cerr << "|| spn_root_maximum:         " << std::get<2>(rootValues) << std::endl;
  std::cerr << "|| spn_root_minimum:         " << std::get<3>(rootValues) << std::endl;
  std::cerr << "|| spn_root_subtree_depth:   " << std::get<4>(rootValues) << std::endl;
  std::cerr << "|| err_rel:                  " << err_rel << std::endl;
  std::cerr << "|| err_abs:                  " << err_abs << std::endl;
  std::cerr << "|| err_log_abs:              " << err_log_abs << std::endl;
  std::cerr << "|| optimal representation:   "; selectedType.dump(); std::cerr << std::endl;
  std::cerr << "==============================================================" << std::endl;

  // std::cerr << "SPNErrorEstimation ctor exit" << std::endl;
}

void SPNErrorEstimation::analyzeGraph(Operation* graphRoot) {
  assert(graphRoot);
  spn_node_values.clear();
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

  estimatedLeastMagnitudeBits = true;
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
          std::cerr << "|| skip_" << iterationCount << std::endl;
          ++iterationCount;
        } else {
          // A fitting format was found, next iteration will start with a format that provides enough magnitude bits.
          break;
        }
      }

      std::cerr << "|| skip?!_" << iterationCount << std::endl;

      // Requirements NOT met.
      satisfied = false;
    }
  }

  if (relative_error == error_model::relative_error) {
    satisfied &= (error_margin > std::abs((value / defect) - 1.0));
    std::cerr << "|| err_rel_" << iterationCount << ":                "<< std::abs((value / defect) - 1.0) << std::endl;
  } else {
    satisfied &= (error_margin > std::abs(value - defect));
    std::cerr << "|| err_abs_" << iterationCount << ":                "<< std::abs(value - defect) << std::endl;
  }

  std::cerr << "|| error_margin_" << iterationCount << ":           "<< error_margin << std::endl;
  std::cerr << "|| satisfied_" << iterationCount << ":              "<< satisfied << std::endl;

  // No further floating point format available -- abort.
  if (!satisfied && (error_model == ERRORMODEL::EM_FLOATING_POINT) && (iterationCount >= (NUM_FLOAT_FORMATS - 1))) {
    abortAnalysis = true;
    selectedType = std::get<2>(Float_Formats[NUM_FLOAT_FORMATS - 1])(rootNode->getContext());
  }

  return satisfied;
}

void SPNErrorEstimation::traverseSubgraph(Operation* subgraphRoot) {
  assert(subgraphRoot);
  auto operands = subgraphRoot->getOperands();

  /*
  std::cerr << " > traverseSubgraph of"  << std::endl;
  subgraphRoot->dump();
  std::cerr << std::endl << " < traverseSubgraph" << std::endl << std::endl;
   */

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
  // std::cerr << "estimateErrorSum entry" << std::endl;
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

  /*
  std::cerr << "s-value:  " << value << std::endl;
  std::cerr << "s-defect: " << defect << std::endl;
  std::cerr << "s-max:    " << max << std::endl;
  std::cerr << "s-min:    " << min << std::endl;
   */

  spn_node_values.emplace(op.getOperation(), std::make_tuple(value, defect, max, min, max_depth));
  // std::cerr << "estimateErrorSum exit" << std::endl;
}

void SPNErrorEstimation::estimateErrorProduct(ProductOp op) {
  // std::cerr << "estimateErrorProduct entry" << std::endl;
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

  /*
  std::cerr << "p-value:  " << value << std::endl;
  std::cerr << "p-defect: " << defect << std::endl;
  std::cerr << "p-max:    " << max << std::endl;
  std::cerr << "p-min:    " << min << std::endl;
   */

  spn_node_values.emplace(op.getOperation(), std::make_tuple(value, defect, max, min, max_depth));
  // std::cerr << "estimateErrorProduct exit" << std::endl;
}

void SPNErrorEstimation::estimateErrorHistogram(HistogramOp op) {
  // std::cerr << "estimateErrorHistogram entry" << std::endl;
  auto buckets = op.buckets().getValue();

  double value;
  double max = std::numeric_limits<double>::min();
  double min = std::numeric_limits<double>::max();
  double defect;

  // op.dump();

  for (auto& b : op.bucketsAttr()) {
    auto val = b.cast<Bucket>().val().getValueAsDouble();;
    max = std::max(max, val);
    min = std::min(min, val);
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

  /*
  std::cerr << "h-value:  " << value << std::endl;
  std::cerr << "h-defect: " << defect << std::endl;
  std::cerr << "h-max:    " << max << std::endl;
  std::cerr << "h-min:    " << min << std::endl;
   */

  spn_node_values.emplace(op.getOperation(), std::make_tuple(value, defect, max, min, 1));
  // std::cerr << "estimateErrorHistogram exit" << std::endl;
}

void SPNErrorEstimation::estimateErrorConstant(ConstantOp op) {
  //std::cerr << "estimateErrorConstant entry" << std::endl;
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

  /*
  std::cerr << "c-value:  " << value << std::endl;
  std::cerr << "c-defect: " << defect << std::endl;
  std::cerr << "c-max:    " << max << std::endl;
  std::cerr << "c-min:    " << min << std::endl;
   */

  spn_node_values.emplace(op.getOperation(), std::make_tuple(value, defect, max, min, 1));
  //std::cerr << "estimateErrorConstant exit" << std::endl;
}

void SPNErrorEstimation::selectOptimalType() {
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
