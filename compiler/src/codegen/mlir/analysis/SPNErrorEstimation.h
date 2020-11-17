//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_ANALYSIS_SPNERRORESTIMATION_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_ANALYSIS_SPNERRORESTIMATION_H

#include <map>
#include <memory>
#include <tuple>
#include <driver/Actions.h>
#include <mlir/IR/Module.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/StandardTypes.h>
#include <util/Logging.h>
#include "SPN/SPNOps.h"

typedef std::shared_ptr<void> arg_t;

using bucket_t = std::tuple<int, int, double>;

namespace mlir {
  namespace spn {

    enum class ERRORMODEL { EM_FIXED_POINT, EM_FLOATING_POINT };

    ///
    /// Class to walk over a (sub-)graph, estimating error margins in the process.
    /// The user provides an error threshold and ... ToDo: Description
    class SPNErrorEstimation {

    public:

      /// Constructor.
      /// \param root Operation pointer which will be treated as SPN-graph root.
      /// \param err_model ERRORMODEL which will determine the considered error model.
      /// \param err_relative Boolean which indicates if relative (true) or absolute (false) error will be considered.
      /// \param err_margin Double which will determine the maximum error-bound.
      explicit SPNErrorEstimation(Operation* root, ERRORMODEL err_model = ERRORMODEL::EM_FLOATING_POINT,
                                  bool err_relative = false, double err_margin = std::numeric_limits<double>::min());

      /// Retrieve the smallest type of the corresponding error model which can represent the SPN result within the
      /// given error margin -OR- if not possible: the biggest type available (e.g. Float64Type).
      /// \return mlir::Type "Optimal" type to represent the given SPN's value(s).
      Type getOptimalType();

    private:

      /// Process provided pointer to a SPN node and update internal counts / results.
      /// \param graphRoot Pointer to the defining operation, representing a SPN node.
      void analyzeGraph(Operation* graphRoot);

      /// Process provided pointer to a SPN node and update internal counts / results.
      /// \param subgraphRoot Pointer to the defining operation, representing a SPN node.
      void traverseSubgraph(Operation* subgraphRoot);

      /// Process minimum and maximum node values to estimate the least amount of magnitude bits.
      void estimateLeastMagnitudeBits();

      /// Check the error requirements .
      /// \return Boolean true if the selected type can represent the SPN's value(s) within the error margin.
      ///                 Otherwise: false.
      bool checkRequirements();

      /// Estimate the error introduced by the given addition operation w.r.t. the current format.
      /// Since the error-estimation needs the "real" value (i.e. exact / accurate), it has to be determined, too.
      // Note: This function will also populate the 'spn_node_maximum' and 'spn_node_minimum'.
      /// \param op Pointer to the defining operation, representing a SPN node.
      /// \return 2-Tuple consisting of (0): the real value and (1): the absolute delta / error.
      void estimateErrorSum(SumOp op);

      /// Estimate the error introduced by the given multiplication operation w.r.t. the current format.
      /// Since the error-estimation needs the "real" value (i.e. exact / accurate), it has to be determined, too.
      // Note: This function will also populate the 'spn_node_maximum' and 'spn_node_minimum'.
      /// \param op Pointer to the defining operation, representing a SPN node.
      /// \return 2-Tuple consisting of (0): the real value and (1): the absolute delta / error.
      void estimateErrorProduct(ProductOp op);

      /// Estimate the error introduced by the given leaf w.r.t. the current format.
      /// Since the error-estimation needs the "real" value (i.e. exact / accurate), it has to be determined, too.
      /// Note: This function will also populate the 'spn_node_maximum' and 'spn_node_minimum'.
      /// \param op Pointer to the defining operation, representing a SPN node.
      /// \return 2-Tuple consisting of (0): the real value and (1): the absolute delta / error.
      void estimateErrorHistogram(HistogramOp op);

      /// Estimate the error introduced by the given constant w.r.t. the current format.
      /// Since the error-estimation needs the "real" value (i.e. exact / accurate), it has to be determined, too.
      // Note: This function will also populate the 'spn_node_maximum' and 'spn_node_minimum'.
      /// \param op Pointer to the defining operation, representing a SPN node.
      /// \return 2-Tuple consisting of (0): the real value and (1): the absolute delta / error.
      void estimateErrorConstant(ConstantOp op);

      /// Select the "optimal" type, using the current state of the analysis.
      void selectOptimalType();

      /// SPN root-node.
      Operation* rootNode;

      /// User-requested error model.
      ERRORMODEL error_model;

      /// User-requested error calculation (true relative error / false: absolute error).
      bool relative_error;

      /// User-requested maximum error margin.
      double error_margin;

      /// Indicate if min-max information was processed.
      int iterationCount = 0;

      /// Indicate if min-max information was processed.
      bool estimatedLeastMagnitudeBits = false;

      /// Indicate if requirements are satisfied.
      bool satisfiedRequirements = false;

      /// Indicate if analysis should be aborted.
      bool abortAnalysis = false;

      /// This is the selected "optimal" mlir::Type.
      Type selectedType;

      /// Number of available floating point formats.
      static const int NUM_FLOAT_FORMATS = 4;

      /// Calculation constant "two".
      const double BASE_TWO = 2.0;

      /// Calculation EPSILON.
      double EPS = 0.0;

      /// Calculation error coefficient, based on EPS.
      double ERR_COEFFICIENT = 1.0;

      /// For each operation store values: { accurate, defective, max, min, max_subtree_depth }
      std::map<mlir::Operation*, std::tuple<double, double, double, double, int>> spn_node_values;

      /// The global extreme-values of the SPN are used when determining the needed I(nteger) / E(xponent) values
      double spn_node_value_global_maximum = std::numeric_limits<double>::min();
      double spn_node_value_global_minimum = std::numeric_limits<double>::max();

      /// Current format information: "Fraction / Mantissa"
      int format_bits_significance = std::numeric_limits<int>::min();
      /// Current format information: "Integer / Exponent"
      int format_bits_magnitude = std::numeric_limits<int>::min();

      /// Tuples which model different floating point formats, like e.g. fp32 / single precision
      /// 0: Significance / "Fraction / Mantissa"
      /// 1: Magnitude / "Integer / Exponent"
      /// 2: mlir::Type
      std::tuple<int, int, Type> Float_Formats[NUM_FLOAT_FORMATS] = {
          {10, 5, Float16Type()},
          {7, 8, BFloat16Type()},
          {23, 8, Float32Type()},
          {52, 11, Float64Type()}
      };

    };

  }

}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_ANALYSIS_SPNERRORESTIMATION_H