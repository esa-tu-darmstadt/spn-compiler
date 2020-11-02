//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_UTIL_ANALYSIS_SPNERRORESTIMATION_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_UTIL_ANALYSIS_SPNERRORESTIMATION_H

#include <codegen/mlir/dialects/spn/SPNDialect.h>
#include <map>
#include <tuple>

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

      /// Default-Constructor.
      explicit SPNErrorEstimation();

      /// Constructor.
      /// \param _root Operation pointer which will be treated as SPN-graph root.
      /// \param _maxError Double which will determine the maximum error-bound.
      /// \param _relativeError Boolean which indicates if relative (true) or absolute (false) error will be considered.
      explicit SPNErrorEstimation(Operation* _root, double _error_margin, bool _relative_error = false,
                                  ERRORMODEL _em = ERRORMODEL::EM_FLOATING_POINT);
    private:

      /// Process provided pointer to a SPN node and update internal counts / results.
      /// \param op Pointer to the defining operation, representing a SPN node.
      /// \param arg Provided state information, e.g. (incremented) level-information from previous calls.
      void visitNode(Operation* op, const arg_t& arg);

      /// Process gathered results ...
      void processResults();

    public:

      /// Update (i.e. re-calculate) ...
      void update();

      /// ToDo: ...
      std::tuple<int,int> determineExactFormat();

      /// ToDo: ...
      int determineExactFormatMantissa(Operation* op);

      /// ToDo: ...
      int determineExactFormatExponent(Operation* op);

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

    private:

      Operation* root;
      bool relative_error;
      double error_margin;
      ERRORMODEL error_model;

      const double BASE_TWO = 2.0;
      double EPS = 0.0;
      double ERR_COEFFICIENT = 1.0;

      // ToDo: Better naming / other datatype.
      std::tuple<double,double> errors[5] = {
          {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}
      };

      /// For each operation store values: { accurate, defective, max, min, max_subtree_depth }
      std::map<Operation*, std::tuple<double,double,double,double,int>> spn_node_values;

      // The global extreme-values of the SPN are used when determining the needed I(nteger) / E(xponent) values
      double spn_node_value_global_maximum = std::numeric_limits<double>::min();
      double spn_node_value_global_minimum = std::numeric_limits<double>::max();

      /// Current format information: "Fraction / Mantissa"
      int format_bits_significance;
      /// Current format information: "Integer / Exponent"
      int format_bits_magnitude;

      // Tuples which model different floating point formats, like e.g. fp32 / single precision
      // First: Mantissa / Second: Exponent
      // Formats? single, double, half, brainfloat16, quad
      std::tuple<int,int,std::string> Float_Formats[5] = {
          {10, 5, "half"}, {7, 8, "bfloat16"}, {23, 8, "single"}, {52, 11, "double"}, {112, 15, "quad"}
      };

    };

  }
}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_UTIL_ANALYSIS_SPNERRORESTIMATION_H