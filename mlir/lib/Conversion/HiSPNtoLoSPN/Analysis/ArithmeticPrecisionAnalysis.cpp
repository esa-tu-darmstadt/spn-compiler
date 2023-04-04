//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include <llvm/Support/Debug.h>
#include <LoSPN/LoSPNAttributes.h>
#include "HiSPNtoLoSPN/ArithmeticPrecisionAnalysis.h"

using namespace mlir::spn;
using namespace mlir::spn::detail;
using namespace mlir::spn::high;

ArithmeticPrecisionAnalysis::ArithmeticPrecisionAnalysis(Operation* rootNode) {
  assert(rootNode);
  llvm::SmallVector<Operation*, 5> queries;
  llvm::SmallVector<Operation*, 5> roots;

  if (auto module = dyn_cast<ModuleOp>(rootNode)) {
    module.walk([&queries, &roots](Operation* op) {
      if (auto query = dyn_cast<QueryInterface>(op)) {
        queries.push_back(op);
      } else if (auto rootNode = dyn_cast<RootNode>(op)) {
        roots.push_back(op);
      }
    });
  } else {
    llvm::dbgs() << "ArithmeticPrecisionAnalysis ctor expected: mlir::ModuleOp\n";
    return;
  }

  if (queries.empty()) {
    llvm::dbgs() << "Did not find any queries to analyze!\n";
    return;
  }

  if (queries.size() > 1) {
    llvm::dbgs() << "Found more than one query, will only analyze the first query!\n";
  }

  auto query = dyn_cast<spn::high::QueryInterface>(queries.front());

  // FIXME: Static data representation should be replaced by extracting this info from somewhere (e.g. the query?).
  est_data_representation = data_representation::EM_FLOATING_POINT;
  est_error_model = query.getErrorModel();
  error_margin = query.getMaxError().convertToDouble();

  // The RootNode acts like an interface, the "desired root" can be obtained by calling root() + getDefiningOp().
  root = dyn_cast<spn::high::RootNode>(roots.front()).getRoot().getDefiningOp();

  iterationCount = 0;
  while (!satisfiedRequirements && !abortAnalysis) {
    analyzeGraph(root);
    ++iterationCount;
  }
  selectOptimalType();
  emitDiagnosticInfo();
}

void ArithmeticPrecisionAnalysis::emitDiagnosticInfo() {
  auto rootValues = spn_node_values[root];
  double value = rootValues.accurate;
  double defect = rootValues.defective;

  double err_rel = std::abs((value / defect) - 1.0);
  double err_abs = std::abs(value - defect);
  double err_abs_log = std::abs(std::log(value / defect));

  std::string emitHeader = "=============== ArithmeticPrecisionAnalysis ===============\n";
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
  rso << "===========================================================\n";

  root->emitRemark("Error Estimation done.").attachNote(root->getLoc()) << rso.str();
}

void ArithmeticPrecisionAnalysis::analyzeGraph(Operation* graphRoot) {
  assert(graphRoot);
  spn_node_values.clear();
  // Either select next floating-point format or increase number of Integer-bits
  format_bits_significance = (est_data_representation == data_representation::EM_FLOATING_POINT) ?
                             Float_Formats[iterationCount].significanceBits : (2 + iterationCount);

  double epsilon = std::pow(BASE_TWO, double(-(format_bits_significance + 1.0)));

  switch (est_data_representation) {
    case data_representation::EM_FLOATING_POINT:
      // Make floating-point calculator.
      calc = std::shared_ptr<ErrorEstimationCalculator>(new detail::FloatingPointCalculator(epsilon));
      break;
    case data_representation::EM_FIXED_POINT:
      // Make fixed-point calculator.
      calc = std::shared_ptr<ErrorEstimationCalculator>(new detail::FixedPointCalculator(epsilon));
      break;
  }

  traverseSubgraph(graphRoot);
  if (!estimatedLeastMagnitudeBits) {
    estimateLeastMagnitudeBits();
  }
  satisfiedRequirements = checkRequirements();
}

void ArithmeticPrecisionAnalysis::estimateLeastMagnitudeBits() {
  // Collect global extreme values
  for (auto entry : spn_node_values) {
    auto value = entry.second;
    spn_node_value_global_maximum = std::max(spn_node_value_global_maximum, value.max);
    spn_node_value_global_minimum = std::min(spn_node_value_global_minimum, value.min);
  }

  format_bits_magnitude = calc->calculateMagnitudeBits(spn_node_value_global_maximum, spn_node_value_global_minimum);
  estimatedLeastMagnitudeBits = true;
}

bool ArithmeticPrecisionAnalysis::checkRequirements() {
  bool satisfied = true;
  auto rootValues = spn_node_values[root];
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
    case mlir::spn::high::error_model::relative_error:err_current = std::abs((value / defect) - 1.0);
      break;
    case mlir::spn::high::error_model::absolute_error:err_current = std::abs(value - defect);
      break;
    case mlir::spn::high::error_model::absolute_log_error:err_current = std::abs(std::log(std::abs(value / defect)));
      break;
  }

  // Check the error margin; but note that the requirements might already be infeasible.
  satisfied &= (error_margin > err_current);

  // No further floating point format available -- abort.
  int maxIndex = (int) Float_Formats.size() - 1;
  if (!satisfied && (est_data_representation == data_representation::EM_FLOATING_POINT)
      && (iterationCount >= maxIndex)) {
    abortAnalysis = true;
    selectedType = Float_Formats.back().getType(root->getContext());
  }

  return satisfied;
}

void ArithmeticPrecisionAnalysis::traverseSubgraph(Operation* subgraphRoot) {
  assert(subgraphRoot);
  if (auto leaf = dyn_cast<mlir::spn::high::LeafNodeInterface>(subgraphRoot)) {
    // Handle leaf node.
    if (auto categorical = dyn_cast<mlir::spn::high::CategoricalNode>(subgraphRoot)) {
      estimateErrorCategorical(categorical);
    } else if (auto gaussian = dyn_cast<mlir::spn::high::GaussianNode>(subgraphRoot)) {
      estimateErrorGaussian(gaussian);
    } else if (auto histogram = dyn_cast<mlir::spn::high::HistogramNode>(subgraphRoot)) {
      estimateErrorHistogram(histogram);
    } else {
      // Unhandled leaf-node-type: Abort.
      llvm::dbgs() << "Encountered unhandled leaf-node-type, operation was: ";
      subgraphRoot->print(llvm::dbgs());
      llvm::dbgs() << "\n";
      assert(false);
    }
  } else {
    // Handle inner node.
    // (1): Visit every child-node.
    auto operands = subgraphRoot->getOperands();
    for (auto child : operands) {
      traverseSubgraph(child.getDefiningOp());
    }

    // (2): Estimate errors of current operation.
    if (auto sum = dyn_cast<mlir::spn::high::SumNode>(subgraphRoot)) {
      estimateErrorSum(sum);
    } else if (auto product = dyn_cast<mlir::spn::high::ProductNode>(subgraphRoot)) {
      estimateErrorProduct(product);
    } else {
      // Unhandled inner node-type: Abort, *should* never happen.
      llvm::dbgs() << "Encountered unhandled inner node-type, operation was: ";
      subgraphRoot->print(llvm::dbgs());
      llvm::dbgs() << "\n";
      assert(false);
    }
  }
}

llvm::SmallVector<ErrorEstimationValue>
    ArithmeticPrecisionAnalysis::estimateErrorBinaryOperation(SmallVector<ErrorEstimationValue> operands, bool isSum) {
  int numOperands = operands.size();

  if (numOperands == 1) {
    // Nothing to do.
    return operands;
  } else if (numOperands == 2) {
    double accurate = 0.0;
    double defective = 0.0;
    double max = 0.0;
    double min = 0.0;
    int depth = 0;

    if (isSum) {
      // Calculate defective sum of two weighted addends.
      defective = calc->calculateDefectiveSum(operands[0], operands[1]);

      // Accumulate accurate values
      accurate = operands[0].accurate + operands[1].accurate;
      depth = std::max(operands[0].depth, operands[1].depth);

      // "max": values will be added
      // "min": the higher value will be ignored entirely (adder becomes min-selection)
      max = operands[0].max + operands[1].max;
      min = std::min(operands[0].min, operands[1].min);
    } else {
      // Calculate defective product value.
      defective = calc->calculateDefectiveProduct(operands[0], operands[1]);

      accurate = operands[0].accurate * operands[1].accurate;
      depth = std::max(operands[0].depth, operands[1].depth);

      // "max": highest values will be multiplied
      // "min": lowest values will be multiplied
      max = operands[0].max * operands[1].max;
      min = operands[0].min * operands[1].min;
    }
    // TODO: Unsure about this
    //operands.set_size(1);
    operands.resize(1);
    operands[0] = ErrorEstimationValue{accurate, defective, max, min, depth};
    return operands;
  } else {
    // Split group of addends.
    auto pivot = llvm::divideCeil(numOperands, 2);
    SmallVector<ErrorEstimationValue> leftOperands;
    SmallVector<ErrorEstimationValue> rightOperands;
    unsigned count = 0;
    for (auto addend : operands) {
      if (count < pivot) {
        leftOperands.push_back(addend);
      } else {
        rightOperands.push_back(addend);
      }
      ++count;
    }

    while (leftOperands.size() > 1) {
      leftOperands = estimateErrorBinaryOperation(leftOperands, isSum);
    }

    while (rightOperands.size() > 1) {
      rightOperands = estimateErrorBinaryOperation(rightOperands, isSum);
    }

    leftOperands.append(rightOperands);
    return estimateErrorBinaryOperation(leftOperands, isSum);
  }
}

void ArithmeticPrecisionAnalysis::estimateErrorSum(mlir::spn::high::SumNode op) {
  // Assumption: There are at least two operands.
  auto operands = op.getOperands();
  assert(operands.size() > 1);
  auto weights = op.getWeights();

  int numOperands = operands.size();
  SmallVector<ErrorEstimationValue> weightedAddends;

  for (int i = 0; i < numOperands; ++i) {
    // Assumption: Operands were encountered beforehand -- i.e. their values are known.
    auto it_v = spn_node_values.find(operands[i].getDefiningOp());
    assert(it_v != spn_node_values.end());
    ErrorEstimationValue& v = it_v->second;

    double weightValue = weights[i].dyn_cast<FloatAttr>().getValueAsDouble();
    double weightDefect = calc->calculateDefectiveLeaf(weightValue);
    ErrorEstimationValue w = ErrorEstimationValue{weightValue, weightDefect, weightValue, weightValue, 1};

    // Treat weighted addend as defective product.
    double defective = calc->calculateDefectiveProduct(v, w);
    double accurate = w.accurate * v.accurate;
    double max = v.max * w.accurate;
    double min = v.min * w.accurate;
    int depth = std::max(v.depth, 1);
    weightedAddends.push_back(ErrorEstimationValue{accurate, defective, max, min, depth});
  }

  auto estimatedSum = estimateErrorBinaryOperation(weightedAddends, true);
  assert(1 == estimatedSum.size());
  auto v = estimatedSum[0];

  // Increase depth, since up until now we only took the operands' depth into account.
  v.depth += 1;
  spn_node_values.emplace(op.getOperation(), v);
}

void ArithmeticPrecisionAnalysis::estimateErrorProduct(mlir::spn::high::ProductNode op) {
  // Assumption: There are at least two operands.
  auto operands = op.getOperands();
  assert(operands.size() > 1);

  // Assumption: Operands were encountered beforehand -- i.e. their values are known.
  auto it_v1 = spn_node_values.find(operands[0].getDefiningOp());
  assert(it_v1 != spn_node_values.end());

  int numOperands = operands.size();
  SmallVector<ErrorEstimationValue> productOperands;

  // Collect all data / operands
  for (int i = 0; i < numOperands; ++i) {
    // Assumption: Operands were encountered beforehand -- i.e. their values are known.
    auto it_v = spn_node_values.find(operands[i].getDefiningOp());
    assert(it_v != spn_node_values.end());
    productOperands.push_back(ErrorEstimationValue{it_v->second});
  }

  auto estimatedProduct = estimateErrorBinaryOperation(productOperands, false);
  assert(1 == estimatedProduct.size());
  auto v = estimatedProduct[0];

  // Increase depth, since up until now we only took the operands' depth into account.
  v.depth += 1;
  spn_node_values.emplace(op.getOperation(), v);
}

void ArithmeticPrecisionAnalysis::estimateErrorCategorical(mlir::spn::high::CategoricalNode op) {
  double value = 0.0;
  double max = std::numeric_limits<double>::min();
  double min = std::numeric_limits<double>::max();
  double defect = 0.0;

  for (auto& p : op.getProbabilitiesAttr().getValue()) {
    double val = p.dyn_cast<FloatAttr>().getValueAsDouble();
    max = std::max(max, val);
    min = std::min(min, val);
  }

  // Use the maximum value
  value = max;
  defect = calc->calculateDefectiveLeaf(value);

  spn_node_values.emplace(op.getOperation(), ErrorEstimationValue{value, defect, max, min, 1});
}

void ArithmeticPrecisionAnalysis::estimateErrorGaussian(mlir::spn::high::GaussianNode op) {
  // Use probability density function (PDF) of the normal distribution
  //   PDF(x) := ( 1 / (stddev * SQRT[2 * Pi] ) ) * EXP( -0.5 * ( [x - mean] / stddev )^2 )
  double stddev = op.getStddev().convertToDouble();
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

void ArithmeticPrecisionAnalysis::estimateErrorHistogram(mlir::spn::high::HistogramNode op) {
  double value = 0.0;
  double max = std::numeric_limits<double>::min();
  double min = std::numeric_limits<double>::max();
  double defect = 0.0;

  for (auto& b : op.getBucketsAttr()) {
    auto val = b.cast<mlir::spn::high::BucketAttr>().getVal().convertToDouble();
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

void ArithmeticPrecisionAnalysis::selectOptimalType() {
  if (!abortAnalysis) {
    switch (est_data_representation) {
      case data_representation::EM_FIXED_POINT:
        selectedType =
            IntegerType::get(root->getContext(), (format_bits_magnitude + format_bits_significance));
        break;
      case data_representation::EM_FLOATING_POINT:
        // Invoke call: Type::get(MLIRContext*)
        selectedType = Float_Formats[iterationCount - 1].getType(root->getContext());
        break;
    }
  }
}

mlir::Type ArithmeticPrecisionAnalysis::getComputationType(bool useLogSpace) {
  // TODO Implement to actually use the analysis results.
  if (useLogSpace) {
    return mlir::spn::low::LogType::get(mlir::FloatType::getF32(root->getContext()));
  }
  return selectedType;
}

inline double mlir::spn::detail::FixedPointCalculator::calculateDefectiveLeaf(double value) {
  return (value + EPS);
}

inline double mlir::spn::detail::FloatingPointCalculator::calculateDefectiveLeaf(double value) {
  return (value * ERR_COEFFICIENT);
}

inline double mlir::spn::detail::FixedPointCalculator::calculateDefectiveProduct(ErrorEstimationValue& v1,
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

inline double mlir::spn::detail::FloatingPointCalculator::calculateDefectiveProduct(ErrorEstimationValue& v1,
                                                                                    ErrorEstimationValue& v2) {
  // Calculate accurate value
  double defect = v1.accurate * v2.accurate;
  // Apply error coefficient for each past conversion step.
  defect *= std::pow(ERR_COEFFICIENT, (v1.depth + v2.depth + 1));
  return defect;
}

inline double mlir::spn::detail::FixedPointCalculator::calculateDefectiveSum(ErrorEstimationValue& v1,
                                                                             ErrorEstimationValue& v2) {
  // Calculate defective value directly
  return (v1.defective + v2.defective);
}

inline double mlir::spn::detail::FloatingPointCalculator::calculateDefectiveSum(ErrorEstimationValue& v1,
                                                                                ErrorEstimationValue& v2) {
  int max_depth = std::max(v1.depth, v2.depth) + 1;
  // Apply error coefficient for the longest conversion chain.
  return ((v1.accurate + v2.accurate) * std::pow(ERR_COEFFICIENT, max_depth));
}

inline int mlir::spn::detail::FixedPointCalculator::calculateMagnitudeBits(double max, double min) {
  return ((int) std::ceil(std::log2(max)));
}

inline int mlir::spn::detail::FloatingPointCalculator::calculateMagnitudeBits(double max, double min) {
  // Overflow and underflow has to be taken into account.
  // Calculate minimum number of bits (then bias) for 2's complement to determine the number of exponent bits.
  double bias = std::ceil(std::abs(std::log2(std::ceil(std::abs(std::log2(min))))));
  bias = std::max(bias, std::ceil(std::abs(std::log2(std::ceil(std::abs(std::log2(max)))))));
  // Calculate actual bias value; to account for special, "unusable" exponent values we do NOT subtract 1.
  bias = std::pow(2.0, bias);
  auto magnitudeBits = (int) std::ceil(std::log2(std::ceil(std::abs(std::log2(min))) + bias));
  magnitudeBits = std::max(magnitudeBits, (int) std::ceil(std::log2(std::ceil(std::abs(std::log2(max))) + bias)));
  return magnitudeBits;
}
