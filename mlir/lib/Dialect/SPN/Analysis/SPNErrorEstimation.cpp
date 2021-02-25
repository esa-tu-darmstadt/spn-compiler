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

  if (auto module = dyn_cast<ModuleOp>(root)) {
    module.walk([&queries, &roots](Operation* op) {
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
  emitDiagnosticInfo();
}

void SPNErrorEstimation::emitDiagnosticInfo() {
  auto rootValues = spn_node_values[rootNode];
  double value = rootValues.accurate;
  double defect = rootValues.defective;

  double err_rel = std::abs((value / defect) - 1.0);
  double err_abs = std::abs(value - defect);
  double err_abs_log = std::abs(std::log(value / defect));

  std::string emitHeader = "===================== SPNErrorEstimation =====================\n";
  llvm::raw_string_ostream rso{emitHeader};
  rso << "|| abortAnalysis:            " << abortAnalysis << "\n";
  rso << "|| satisfiedRequirements:    " << satisfiedRequirements << "\n";
  rso << "|| iterationCount:           " << iterationCount << "\n";
  rso << "|| format_bits_magnitude:    " << format_bits_magnitude << "\n";
  rso << "|| format_bits_significance: " << format_bits_significance << "\n";
  rso << "|| EPS:                      " << calc->getEpsilon() << "\n";
  rso << "|| ERR_COEFFICIENT:          " << calc->getErrorCoefficient() << "\n";
  rso << "|| err_margin:               " << error_margin << "\n";
  rso << "|| spn_node_global_maximum:  " << spn_node_value_global_maximum << "\n";
  rso << "|| spn_node_global_minimum:  " << spn_node_value_global_minimum << "\n";
  rso << "|| spn_root_value:           " << rootValues.accurate << "\n";
  rso << "|| spn_root_defect:          " << rootValues.defective << "\n";
  rso << "|| spn_root_maximum:         " << rootValues.max << "\n";
  rso << "|| spn_root_minimum:         " << rootValues.min << "\n";
  rso << "|| spn_root_subtree_depth:   " << rootValues.depth << "\n";
  rso << "|| err_rel:                  " << err_rel << "\n";
  rso << "|| err_abs:                  " << err_abs << "\n";
  rso << "|| err_log_abs:              " << err_abs_log << "\n";
  rso << "|| optimal representation:   ";
  selectedType.print(rso);
  rso << "\n";
  rso << "==============================================================\n";

  rootNode->emitRemark("Error Estimation done.").attachNote(rootNode->getLoc()) << rso.str();
}

void SPNErrorEstimation::analyzeGraph(Operation* graphRoot) {
  assert(graphRoot);
  spn_node_values.clear();
  // Either select next floating-point format or increase number of Integer-bits
  format_bits_significance = (est_data_representation == data_representation::EM_FLOATING_POINT) ?
                             Float_Formats[iterationCount].significanceBits : (2 + iterationCount);

  double epsilon = std::pow(BASE_TWO, double(-(format_bits_significance + 1.0)));

  switch (est_data_representation) {
    case data_representation::EM_FLOATING_POINT:
      // Make floating-point calculator.
      calc = std::shared_ptr<ErrorEstimationCalculator>(new FloatingPointCalculator(epsilon));
      break;
    case data_representation::EM_FIXED_POINT:
      // Make fixed-point calculator.
      calc = std::shared_ptr<ErrorEstimationCalculator>(new FixedPointCalculator(epsilon));
      break;
  }

  traverseSubgraph(graphRoot);
  if (!estimatedLeastMagnitudeBits) {
    estimateLeastMagnitudeBits();
  }
  satisfiedRequirements = checkRequirements();
}

void SPNErrorEstimation::estimateLeastMagnitudeBits() {
  // Find global extreme values
  for (auto entry : spn_node_values) {
    auto value = entry.second;
    // ToDo: Question what's the desired behavior in cases like:
    //  (spn_node_value_global_maximum == 0.0) -AND/OR- (spn_node_value_global_minimum == 1.0)?
    // ToDo: RFC on filtering out these special cases.
    if (value.max < 1.0) {
      spn_node_value_global_maximum = std::max(spn_node_value_global_maximum, value.max);
    }
    if (value.min > 0.0) {
      spn_node_value_global_minimum = std::min(spn_node_value_global_minimum, value.min);
    }
  }

  format_bits_magnitude = calc->calculateMagnitudeBits(spn_node_value_global_maximum, spn_node_value_global_minimum);
  estimatedLeastMagnitudeBits = true;
}

bool SPNErrorEstimation::checkRequirements() {
  bool satisfied = true;
  auto rootValues = spn_node_values[rootNode];
  double value = rootValues.accurate;
  double defect = rootValues.defective;

  // Floating-point: Check if currently selected format has enough magnitude bits
  if (est_data_representation == data_representation::EM_FLOATING_POINT) {
    // If least amount of magnitude bits is greater than currently selected format's: Search for fitting format.
    if (format_bits_magnitude > Float_Formats[iterationCount].magnitudeBits) {
      // Requirements NOT met.
      satisfied = false;

      for (int i = iterationCount + 1; i < (int) Float_Formats.size(); ++i) {
        if (format_bits_magnitude > Float_Formats[i].magnitudeBits) {
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
    case error_model::relative_error:err_current = std::abs((value / defect) - 1.0);
      break;
    case error_model::absolute_error:err_current = std::abs(value - defect);
      break;
    case error_model::absolute_log_error:err_current = std::abs(std::log(std::abs(value / defect)));
      break;
  }

  // Check the error margin; but note that the requirements might already be infeasible.
  satisfied &= (error_margin > err_current);

  // No further floating point format available -- abort.
  int maxIndex = (int) Float_Formats.size() - 1;
  if (!satisfied && (est_data_representation == data_representation::EM_FLOATING_POINT)
      && (iterationCount >= maxIndex)) {
    abortAnalysis = true;
    selectedType = Float_Formats.back().getType(rootNode->getContext());
  }

  return satisfied;
}

void SPNErrorEstimation::traverseSubgraph(Operation* subgraphRoot) {
  assert(subgraphRoot);

  if (auto leaf = dyn_cast<LeafNodeInterface>(subgraphRoot)) {
    // Handle leaf node.
    if (auto categorical = dyn_cast<CategoricalOp>(subgraphRoot)) {
      estimateErrorCategorical(categorical);
    } else if (auto gaussian = dyn_cast<GaussianOp>(subgraphRoot)) {
      estimateErrorGaussian(gaussian);
    } else if (auto histogram = dyn_cast<HistogramOp>(subgraphRoot)) {
      estimateErrorHistogram(histogram);
    } else {
      // Unhandled leaf-node-type: Abort.
      llvm::dbgs() << "Encountered unhandled leaf-node-type, operation was: ";
      subgraphRoot->print(llvm::dbgs());
      llvm::dbgs() << "\n";
      assert(false);
    }
  } else if (auto constant = dyn_cast<ConstantOp>(subgraphRoot)) {
    // Handle constant node.
    estimateErrorConstant(constant);
  } else {
    // Handle inner node.
    arg_t passed_arg(nullptr);

    // First: Visit every child-node.
    auto operands = subgraphRoot->getOperands();
    for (auto child : operands) {
      traverseSubgraph(child.getDefiningOp());
    }

    // Second: Estimate errors of current operation.
    if (auto sum = dyn_cast<SumOp>(subgraphRoot)) {
      estimateErrorSum(sum);
    } else if (auto product = dyn_cast<ProductOp>(subgraphRoot)) {
      estimateErrorProduct(product);
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

  // Retrieve the corresponding data
  ErrorEstimationValue& v1 = it_v1->second;
  ErrorEstimationValue& v2 = it_v2->second;

  value = v1.accurate + v2.accurate;
  max_depth = std::max(v1.depth, v2.depth) + 1;
  defect = calc->calculateDefectiveSum(v1, v2);

  // "max": values will be added
  // "min": the higher value will be ignored entirely (adder becomes min-selection)
  max = v1.max + v2.max;
  min = std::min(v1.min, v2.min);

  spn_node_values.emplace(op.getOperation(), ErrorEstimationValue{value, defect, max, min, max_depth});
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

  // Retrieve the corresponding data
  ErrorEstimationValue& v1 = it_v1->second;
  ErrorEstimationValue& v2 = it_v2->second;

  value = v1.accurate * v2.accurate;
  max_depth = std::max(v1.depth, v2.depth) + 1;
  defect = calc->calculateDefectiveProduct(v1, v2);

  // "max": highest values will be multiplied
  // "min": lowest values will be multiplied
  max = v1.max * v2.max;
  min = v1.min * v2.min;

  spn_node_values.emplace(op.getOperation(), ErrorEstimationValue{value, defect, max, min, max_depth});
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
  defect = calc->calculateDefectiveLeaf(value);

  spn_node_values.emplace(op.getOperation(), ErrorEstimationValue{value, defect, max, min, 1});
}

void SPNErrorEstimation::estimateErrorConstant(ConstantOp op) {
  double value = op.value().convertToDouble();
  double max = value;
  double min = value;
  double defect = calc->calculateDefectiveLeaf(value);

  spn_node_values.emplace(op.getOperation(), ErrorEstimationValue{value, defect, max, min, 1});
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
  double defect = calc->calculateDefectiveLeaf(value);

  // Consider 99%, i.e. x := mean +/- (2.576 * stddev)
  // Note 1: mean will be subtracted upon calculating the PDF; hence omit it in the first place: x := 2.576 * stddev
  // Note 2: x will be divided by stddev leaving only constant ~ 2.576.
  // No skew -> PDF is symmetric w.r.t. to PDF(mean), so PDF(mean + delta) = PDF(mean - delta)
  double min = c * GAUSS_99;

  spn_node_values.emplace(op.getOperation(), ErrorEstimationValue{value, defect, max, min, 1});
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
  defect = calc->calculateDefectiveLeaf(value);

  // Check for changed values.
  assert(max != std::numeric_limits<double>::min());
  assert(min != std::numeric_limits<double>::max());

  spn_node_values.emplace(op.getOperation(), ErrorEstimationValue{value, defect, max, min, 1});
}

void SPNErrorEstimation::selectOptimalType() {
  if (!abortAnalysis) {
    switch (est_data_representation) {
      case data_representation::EM_FIXED_POINT:
        selectedType =
            IntegerType::get(rootNode->getContext(), (format_bits_magnitude + format_bits_significance));
        break;
      case data_representation::EM_FLOATING_POINT:
        // Invoke call: Type::get(MLIRContext*)
        selectedType = Float_Formats[iterationCount - 1].getType(rootNode->getContext());
        break;
    }
  }
}

mlir::Type SPNErrorEstimation::getOptimalType() {
  return selectedType;
}

inline double SPNErrorEstimation::FixedPointCalculator::calculateDefectiveLeaf(double value) {
  return (value + EPS);
}

inline double SPNErrorEstimation::FloatingPointCalculator::calculateDefectiveLeaf(double value) {
  return (value * ERR_COEFFICIENT);
}

inline double SPNErrorEstimation::FixedPointCalculator::calculateDefectiveProduct(ErrorEstimationValue& v1,
                                                                                  ErrorEstimationValue& v2) {
  // Calculate accurate value
  double defect = v1.accurate * v2.accurate;
  // Calculate deltas of the given operators (see equation (5) of ProbLP paper)
  double delta_t1 = v1.defective - v1.accurate;
  double delta_t2 = v2.defective - v2.accurate;
  // Add the corresponding total delta to the already calculated ("accurate") value
  defect += (v1.max * delta_t2) + (v2.max * delta_t1) + (delta_t1 * delta_t2) + EPS;
  return defect;
}

inline double SPNErrorEstimation::FloatingPointCalculator::calculateDefectiveProduct(ErrorEstimationValue& v1,
                                                                                     ErrorEstimationValue& v2) {
  // Calculate accurate value
  double defect = v1.accurate * v2.accurate;
  // Apply error coefficient for each past conversion step.
  defect *= std::pow(ERR_COEFFICIENT, (v1.depth + v2.depth + 1));
  return defect;
}

inline double SPNErrorEstimation::FixedPointCalculator::calculateDefectiveSum(ErrorEstimationValue& v1,
                                                                              ErrorEstimationValue& v2) {
  // Calculate defective value directly
  return (v1.defective + v2.defective);
}

inline double SPNErrorEstimation::FloatingPointCalculator::calculateDefectiveSum(ErrorEstimationValue& v1,
                                                                                 ErrorEstimationValue& v2) {
  int max_depth = std::max(v1.depth, v2.depth) + 1;
  // Apply error coefficient for the longest conversion chain.
  return ((v1.accurate + v2.accurate) * std::pow(ERR_COEFFICIENT, max_depth));
}

inline int SPNErrorEstimation::FixedPointCalculator::calculateMagnitudeBits(double max, double min) {
  return ((int) std::ceil(std::log2(max)));
}

inline int SPNErrorEstimation::FloatingPointCalculator::calculateMagnitudeBits(double max, double min) {
  // Overflow and underflow has to be taken into account.
  // Calculate minimum number of bits (then bias) for 2's complement to determine the number of exponent bits.
  double bias = std::ceil(std::abs(std::log2(std::ceil(std::abs(std::log2(min))))));
  bias = std::max(bias, std::ceil(std::abs(std::log2(std::ceil(std::abs(std::log2(max)))))));
  // Calculate actual bias value; to account for special, "unusable" exponent values we do NOT subtract 1.
  bias = std::pow(2.0, bias);
  int magnitudeBits = (int) std::ceil(std::log2(std::ceil(std::abs(std::log2(min))) + bias));
  magnitudeBits = std::max(magnitudeBits, (int) std::ceil(std::log2(std::ceil(std::abs(std::log2(max))) + bias)));
  return magnitudeBits;
}
