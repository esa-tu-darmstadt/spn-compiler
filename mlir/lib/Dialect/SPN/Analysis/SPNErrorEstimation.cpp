//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <llvm/Support/Debug.h>
#include <SPN/SPNAttributes.h>
#include "SPN/Analysis/SPNErrorEstimation.h"

using namespace mlir::spn;

SPNErrorEstimation::SPNErrorEstimation(Operation* root) {
  assert(root);
  llvm::SmallVector<Operation*, 5> queries;
  llvm::SmallVector<Operation*, 5> roots;

  if (auto module = dyn_cast<mlir::ModuleOp>(root)) {
    module.walk([&queries,&roots](Operation* op) {
      if (auto query = dyn_cast<QueryInterface>(op)) {
        queries.push_back(op);
        for (auto r : query.getRootNodes()) {
          roots.push_back(r->getOperand(0).getDefiningOp());
        }
      }
    });
  } else {
    llvm::dbgs() << "SPNErrorEstimation ctor expected: mlir::ModuleOp\n";
    return;
  }

  if (queries.empty()) {
    llvm::dbgs() << "Did not find any queries to analyze!\n";
    return;
  }

  if (queries.size() > 1) {
    llvm::dbgs() << "Found more than one query, will only analyze the first query!\n";
  }

  auto query = dyn_cast<QueryInterface>(queries.front());

  rootNode = roots.front();
  // FIXME: Static data representation should be replaced by extracting this info from somewhere (e.g. the query?).
  est_data_representation = data_representation::EM_FLOATING_POINT;
  est_error_model = query.getErrorModel();
  error_margin = query.getMaxError();

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
  double err_abs_log = std::log((1.0 + std::abs((value / defect) - 1.0)));

  llvm::dbgs() << "===================== SPNErrorEstimation =====================\n";
  llvm::dbgs() << "|| abortAnalysis:            " << abortAnalysis << "\n";
  llvm::dbgs() << "|| satisfiedRequirements:    " << satisfiedRequirements << "\n";
  llvm::dbgs() << "|| iterationCount:           " << iterationCount << "\n";
  llvm::dbgs() << "|| format_bits_magnitude:    " << format_bits_magnitude << "\n";
  llvm::dbgs() << "|| format_bits_significance: " << format_bits_significance << "\n";
  llvm::dbgs() << "|| EPS:                      " << EPS << "\n";
  llvm::dbgs() << "|| ERR_COEFFICIENT:          " << ERR_COEFFICIENT << "\n";
  llvm::dbgs() << "|| err_margin:               " << error_margin << "\n";
  llvm::dbgs() << "|| spn_node_global_maximum:  " << spn_node_value_global_maximum << "\n";
  llvm::dbgs() << "|| spn_node_global_minimum:  " << spn_node_value_global_minimum << "\n";
  llvm::dbgs() << "|| spn_root_value:           " << std::get<0>(rootValues) << "\n";
  llvm::dbgs() << "|| spn_root_defect:          " << std::get<1>(rootValues) << "\n";
  llvm::dbgs() << "|| spn_root_maximum:         " << std::get<2>(rootValues) << "\n";
  llvm::dbgs() << "|| spn_root_minimum:         " << std::get<3>(rootValues) << "\n";
  llvm::dbgs() << "|| spn_root_subtree_depth:   " << std::get<4>(rootValues) << "\n";
  llvm::dbgs() << "|| err_rel:                  " << err_rel << "\n";
  llvm::dbgs() << "|| err_abs:                  " << err_abs << "\n";
  llvm::dbgs() << "|| err_log_abs:              " << err_abs_log << "\n";
  llvm::dbgs() << "|| optimal representation:   ";
  selectedType.dump();
  llvm::dbgs() << "\n";
  llvm::dbgs() << "==============================================================\n";
}

void SPNErrorEstimation::analyzeGraph(Operation* graphRoot) {
  assert(graphRoot);
  spn_node_values.clear();
  // Either select next floating-point format or increase number of Integer-bits
  format_bits_significance = (est_data_representation == data_representation::EM_FLOATING_POINT) ?
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
    // ToDo: Question what's the desired behavior in cases like:
    //  (spn_node_value_global_maximum == 0.0) -AND/OR- (spn_node_value_global_minimum == 1.0)?
    // ToDo: RFC on filtering out these special cases.
    if (std::get<2>(entry.second) < 1.0) {
      spn_node_value_global_maximum = std::max(spn_node_value_global_maximum, std::get<2>(entry.second));
    }
    if (std::get<3>(entry.second) > 0.0) {
      spn_node_value_global_minimum = std::min(spn_node_value_global_minimum, std::get<3>(entry.second));
    }
  }

  double bias;

  switch (est_data_representation) {
    case data_representation::EM_FLOATING_POINT:
      // Overflow and underflow has to be taken into account.
      // Calculate bias for 2's complement to determine the number of exponent bits.
      bias = std::ceil(std::abs(std::log2(std::ceil(std::abs(std::log2(spn_node_value_global_minimum))))));
      bias = std::max(bias, std::abs(std::log2(std::ceil(std::ceil(std::abs(std::log2(spn_node_value_global_maximum)))))));
      // Calculate actual bias value; to account for special, "unusable" exponent values we do NOT subtract 1.
      bias = std::pow(2.0, bias);
      format_bits_magnitude = (int) std::ceil(std::log2(std::abs(std::log2(spn_node_value_global_minimum)) + bias));
      format_bits_magnitude =
          std::max(format_bits_magnitude,
                   (int) std::ceil(std::log2(std::ceil(std::abs(std::log2(spn_node_value_global_maximum))) + bias)));
      break;
    case data_representation::EM_FIXED_POINT:
      // Only overflow / maximum-value has to be taken into account for integer bits.
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
  if (est_data_representation == data_representation::EM_FLOATING_POINT) {
    // If least amount of magnitude bits is greater than currently selected format's: Search for fitting format.
    if (format_bits_magnitude > std::get<1>(Float_Formats[iterationCount])) {
      // Requirements NOT met.
      satisfied = false;

      for (int i = iterationCount + 1; i < NUM_FLOAT_FORMATS; ++i) {
        if (format_bits_magnitude > std::get<1>(Float_Formats[i])) {
          // Each time we check a NON-fitting format we can skip the corresponding iteration.
          ++iterationCount;
        } else {
          // A fitting format was found, next iteration will start with a format that provides enough magnitude bits.
          break;
        }
      }
    }
  }

  double err_current = std::numeric_limits<double>::infinity();

  switch (est_error_model) {
    case error_model::relative_error:
      err_current = std::abs((value / defect) - 1.0);
      break;
    case error_model::absolute_error:
      err_current = std::abs(value - defect);
      break;
    case error_model::absolute_log_error:
      err_current = std::abs(std::log(std::abs(value / defect)));
      break;
  }

  // Check the error margin; but note that the requirements might already be infeasible.
  satisfied &= (error_margin > err_current);

  // No further floating point format available -- abort.
  if (!satisfied && (est_data_representation == data_representation::EM_FLOATING_POINT) && (iterationCount >= (NUM_FLOAT_FORMATS - 1))) {
    abortAnalysis = true;
    selectedType = std::get<2>(Float_Formats[NUM_FLOAT_FORMATS - 1])(rootNode->getContext());
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
    if (auto categorical = dyn_cast<CategoricalOp>(subgraphRoot)) {
      estimateErrorCategorical(categorical);
    } else if (auto gaussian = dyn_cast<GaussianOp>(subgraphRoot)) {
      estimateErrorGaussian(gaussian);
    } else if (auto histogram = dyn_cast<HistogramOp>(subgraphRoot)) {
      estimateErrorHistogram(histogram);
    } else {
      // Unhandled leaf-node-type: Abort.
      llvm::dbgs() << "Encountered unhandled leaf-node-type: ";
      subgraphRoot->dump();
      llvm::dbgs() << "\n";
      assert(false);
    }
  } else {
    if (auto constant = dyn_cast<ConstantOp>(subgraphRoot)) {
      estimateErrorConstant(constant);
    }
  }
}

void SPNErrorEstimation::estimateErrorSum(SumOp op) {
  double value, defect, max, min;
  int max_depth;

  // Assumption: There are exactly & guaranteed(!) two operands
  auto operands = op.getOperands();
  assert(operands.size() == 2);

  // Assumption: Both operands were encountered beforehand -- i.e. their values are known.
  auto it_v1 = spn_node_values.find(operands[0].getDefiningOp());
  auto it_v2 = spn_node_values.find(operands[1].getDefiningOp());
  assert(it_v1 != spn_node_values.end());
  assert(it_v2 != spn_node_values.end());

  // Retrieve the corresponding tuples
  std::tuple<double, double, double, double, int>& t1 = it_v1->second;
  std::tuple<double, double, double, double, int>& t2 = it_v2->second;

  value = std::get<0>(t1) + std::get<0>(t2);

  max_depth = std::max(std::get<4>(t1), std::get<4>(t2)) + 1;

  switch (est_data_representation) {
    case data_representation::EM_FIXED_POINT:defect = std::get<1>(t1) + std::get<1>(t2);
      break;
    case data_representation::EM_FLOATING_POINT:defect = value * std::pow(ERR_COEFFICIENT, max_depth);
      break;
  }

  // "max": values will be added
  // "min": the higher value will be ignored entirely (adder becomes min-selection)
  max = std::get<2>(t1) + std::get<2>(t2);
  min = std::min(std::get<3>(t1), std::get<3>(t2));

  spn_node_values.emplace(op.getOperation(), std::make_tuple(value, defect, max, min, max_depth));
}

void SPNErrorEstimation::estimateErrorProduct(ProductOp op) {
  double value, defect, max, min;
  int max_depth;

  // Assumption: There are exactly & guaranteed(!) two operands
  auto operands = op.getOperands();
  assert(operands.size() == 2);

  // Assumption: Both operands were encountered beforehand -- i.e. their values are known.
  auto it_v1 = spn_node_values.find(operands[0].getDefiningOp());
  auto it_v2 = spn_node_values.find(operands[1].getDefiningOp());
  assert(it_v1 != spn_node_values.end());
  assert(it_v2 != spn_node_values.end());

  // Retrieve the corresponding tuples
  std::tuple<double, double, double, double, int>& t1 = it_v1->second;
  std::tuple<double, double, double, double, int>& t2 = it_v2->second;

  value = std::get<0>(t1) * std::get<0>(t2);
  defect = value;

  max_depth = std::max(std::get<4>(t1), std::get<4>(t2)) + 1;

  // Moved declaration(s) out of switch-case to avoid errors / warnings.
  double delta_t1, delta_t2;
  int accumulated_depth;

  switch (est_data_representation) {
    case data_representation::EM_FIXED_POINT:
      // Calculate deltas of the given operators (see equation (5) of ProbLP paper)
      delta_t1 = std::get<1>(t1) - std::get<0>(t1);
      delta_t2 = std::get<1>(t2) - std::get<0>(t2);

      // Add the corresponding total delta to the already calculated ("accurate") value
      defect += (std::get<2>(t1) * delta_t2) + (std::get<2>(t2) * delta_t1) + (delta_t1 * delta_t2) + EPS;
      break;
    case data_representation::EM_FLOATING_POINT:accumulated_depth = std::get<4>(t1) + std::get<4>(t2) + 1;
      defect *= std::pow(ERR_COEFFICIENT, accumulated_depth);
      break;
  }

  // "max": highest values will be multiplied
  // "min": lowest values will be multiplied
  max = std::get<2>(t1) * std::get<2>(t2);
  min = std::get<3>(t1) * std::get<3>(t2);

  spn_node_values.emplace(op.getOperation(), std::make_tuple(value, defect, max, min, max_depth));
}

void SPNErrorEstimation::estimateErrorCategorical(CategoricalOp op) {
  double value, max, min, defect;

  for (auto& p : op.probabilitiesAttr().getValue()) {
    double val = p.dyn_cast<FloatAttr>().getValueAsDouble();
    max = std::max(max, val);
    min = std::min(min, val);
  }

  // Use the maximum value
  value = max;

  switch (est_data_representation) {
    case data_representation::EM_FIXED_POINT:defect = value + EPS;
      break;
    case data_representation::EM_FLOATING_POINT:defect = value * ERR_COEFFICIENT;
      break;
  }

  spn_node_values.emplace(op.getOperation(), std::make_tuple(value, defect, max, min, 1));
}

void SPNErrorEstimation::estimateErrorConstant(ConstantOp op) {
  double value = op.value().convertToDouble();
  double max = value;
  double min = value;
  double defect = 0.0;

  switch (est_data_representation) {
    case data_representation::EM_FIXED_POINT:defect = value + EPS;
      break;
    case data_representation::EM_FLOATING_POINT:defect = value * ERR_COEFFICIENT;
      break;
  }

  spn_node_values.emplace(op.getOperation(), std::make_tuple(value, defect, max, min, 1));
}

void SPNErrorEstimation::estimateErrorGaussian(GaussianOp op) {
  // Use probability density function (PDF) of the normal distribution
  //   PDF(x) := ( 1 / (stddev * SQRT[2 * Pi] ) ) * EXP( -0.5 * ( [x - mean] / stddev )^2 )
  double stddev = op.stddev().convertToDouble();
  // Define c := ( 1 / (stddev * SQRT[2 * Pi] ) )
  double c = std::pow((stddev * std::sqrt(2 * M_PI)), -1.0);

  // Since we do not treat skewed distributions, set value as PDF(mean), which is also the maximum.
  // value := PDF(mean) = c * EXP(0) = c
  double value = c;
  double max = value;

  // Consider 99%, i.e. x := mean +/- (2.576 * stddev)
  // Note 1: mean will be subtracted upon calculating the PDF; hence omit it in the first place: x := 2.576 * stddev
  // Note 2: x will be divided by stddev leaving only constant ~ 2.576.
  // No skew -> PDF is symmetric w.r.t. to PDF(mean), so PDF(mean + delta) = PDF(mean - delta)
  double min = c * GAUSS_99;
  double defect;

  switch (est_data_representation) {
    case data_representation::EM_FIXED_POINT:defect = value + EPS;
      break;
    case data_representation::EM_FLOATING_POINT:defect = value * ERR_COEFFICIENT;
      break;
  }

  spn_node_values.emplace(op.getOperation(), std::make_tuple(value, defect, max, min, 1));
}

void SPNErrorEstimation::estimateErrorHistogram(HistogramOp op) {
  double value;
  double max = std::numeric_limits<double>::min();
  double min = std::numeric_limits<double>::max();
  double defect;

  for (auto& b : op.bucketsAttr()) {
    auto val = b.cast<Bucket>().val().getValueAsDouble();
    max = std::max(max, val);
    min = std::min(min, val);
  }

  // Select maximum as "value"
  value = max;

  switch (est_data_representation) {
    case data_representation::EM_FIXED_POINT:defect = value + EPS;
      break;
    case data_representation::EM_FLOATING_POINT:defect = value * ERR_COEFFICIENT;
      break;
  }

  // Check for changed values.
  assert(max != std::numeric_limits<double>::min());
  assert(min != std::numeric_limits<double>::max());

  spn_node_values.emplace(op.getOperation(), std::make_tuple(value, defect, max, min, 1));
}

void SPNErrorEstimation::selectOptimalType() {
  if (!abortAnalysis) {
    switch (est_data_representation) {
      case data_representation::EM_FIXED_POINT:
        selectedType =
            IntegerType::get((format_bits_magnitude + format_bits_significance), rootNode->getContext());
        break;
      case data_representation::EM_FLOATING_POINT:
        // Invoke call: Type::get(MLIRContext*)
        selectedType = std::get<2>(Float_Formats[iterationCount - 1])(rootNode->getContext());
        break;
    }
  }
}

mlir::Type SPNErrorEstimation::getOptimalType() {
  return selectedType;
}
