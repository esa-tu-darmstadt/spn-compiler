//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_HISPNTOLOSPN_ARITHMETICPRECISIONANALYSIS_H
#define SPNC_MLIR_INCLUDE_CONVERSION_HISPNTOLOSPN_ARITHMETICPRECISIONANALYSIS_H

#include <map>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "HiSPN/HiSPNOps.h"
#include "HiSPN/HiSPNAttributes.h"
#include "LoSPN/LoSPNTypes.h"

namespace mlir {
  namespace spn {
    namespace detail {
      /// This struct allows to store different value characteristics: accurate-, defective-, max-, min-value and depth.
      struct ErrorEstimationValue {
        /// Accurate value.
        double accurate;
        /// Defective value. (Accurate + delta)
        double defective;
        /// Maximum value.
        double max;
        /// Minimum value.
        double min;
        /// Maximum sub-tree depth of the corresponding (SPN-)value.
        int depth;
      };

      /// Struct which models different data formats, like e.g. fp32 / single precision: {8, 23, Float32Type::get}
      struct ValueFormat {
        /// Number of magnitude bits, i.e. "Integer / Exponent"
        int magnitudeBits;
        /// Number of significance bits, i.e. "Fraction / Mantissa"
        int significanceBits;
        /// Function returning mlir::Type, requiring an MLIRContext*
        std::function<Type(MLIRContext*)> getType;
      };

      /// Class which acts as Strategy (GoF pattern) super-class for the error calculation of different data formats.
      class ErrorEstimationCalculator {
      public:
        explicit ErrorEstimationCalculator(double epsilon = 0.0) : EPS(epsilon), ERR_COEFFICIENT(1.0 + epsilon) {}
        virtual int calculateMagnitudeBits(double max, double min) = 0;
        virtual double calculateDefectiveLeaf(double value) = 0;
        virtual double calculateDefectiveProduct(ErrorEstimationValue& v1, ErrorEstimationValue& v2) = 0;
        virtual double calculateDefectiveSum(ErrorEstimationValue& v1, ErrorEstimationValue& v2) = 0;
        virtual ~ErrorEstimationCalculator() = default;
        double getEpsilon() const { return EPS; }
        double getErrorCoefficient() const { return ERR_COEFFICIENT; }
      protected:
        /// Calculation EPSILON.
        double EPS;
        /// Calculation error coefficient: 1 + EPSILON.
        double ERR_COEFFICIENT;
      };

      /// Class implementing the actual error calculations (defective values) regarding the fixed point format.
      class FixedPointCalculator : public ErrorEstimationCalculator {
      public:
        explicit FixedPointCalculator(double _eps) : ErrorEstimationCalculator(_eps) {}
        double calculateDefectiveLeaf(double value) override;
        double calculateDefectiveProduct(ErrorEstimationValue& v1, ErrorEstimationValue& v2) override;
        double calculateDefectiveSum(ErrorEstimationValue& v1, ErrorEstimationValue& v2) override;
        int calculateMagnitudeBits(double max, double min) override;
      };

      /// Class implementing the actual error calculations (defective values) regarding the floating point format.
      class FloatingPointCalculator : public ErrorEstimationCalculator {
      public:
        explicit FloatingPointCalculator(double _eps) : ErrorEstimationCalculator(_eps) {}
        double calculateDefectiveLeaf(double value) override;
        double calculateDefectiveProduct(ErrorEstimationValue& v1, ErrorEstimationValue& v2) override;
        double calculateDefectiveSum(ErrorEstimationValue& v1, ErrorEstimationValue& v2) override;
        int calculateMagnitudeBits(double max, double min) override;
      };
    }
  }
}

namespace mlir {
  namespace spn {

    enum class data_representation { EM_FIXED_POINT, EM_FLOATING_POINT };

    ///
    /// Analysis performing an arithmetic error estimation
    /// on a graph to determine the best datatype to use
    /// for computation.
    /// Note: Currently, the data representation is fixed to: floating point.
    class ArithmeticPrecisionAnalysis {

    public:

      ArithmeticPrecisionAnalysis()
          : est_data_representation(data_representation::EM_FLOATING_POINT),
            est_error_model(mlir::spn::high::error_model::absolute_error), calc(nullptr) {}

      /// Constructor. Note that this will be primarily used as an analysis.
      /// \param root Operation pointer which will be treated as SPN-graph root.
      explicit ArithmeticPrecisionAnalysis(Operation* root);

      /// Retrieve the smallest type of the corresponding error model which can represent the SPN result within the
      /// given error margin -OR- if not possible: the biggest type available (e.g. Float64Type).
      /// \param bool True if computation is done within log-space.
      /// \return mlir::Type "Optimal" type to represent the given SPN's value(s).
      mlir::Type getComputationType(bool useLogSpace);

    private:

      /// Process provided pointer to a SPN node and update internal counts / results.
      /// \param graphRoot Pointer to the defining operation, representing a SPN node.
      void analyzeGraph(Operation* graphRoot);

      /// Process provided pointer to a SPN node and update internal counts / results.
      /// \param subgraphRoot Pointer to the defining operation, representing a SPN node.
      void traverseSubgraph(Operation* subgraphRoot);

      /// Process minimum and maximum node values to estimate the least amount of magnitude (Integer / Exponent) bits.
      void estimateLeastMagnitudeBits();

      /// Check the error requirements.
      /// \return Boolean true if the selected type can represent the SPN's value(s) within the error margin.
      ///                 Otherwise: false.
      bool checkRequirements();

      /// Estimate error of Sum or Product operations, in a binary-tree-like evaluation order.
      /// \param operands SmallVector<ErrorEstimationValue> The corresponding operation's operands (e.g. addends).
      /// \param isSum Boolean If set to true will treat operands as addends, otherwise as multiplicands.
      /// \return SmallVector<ErrorEstimationValue> with a single element, representing the whole Sum or Product.
      llvm::SmallVector<spn::detail::ErrorEstimationValue>
        estimateErrorBinaryOperation(SmallVector<spn::detail::ErrorEstimationValue> operands, bool isSum);

      /// Estimate the error introduced by the given (weighted) addition operation w.r.t. the current format.
      /// \param op Pointer to the defining operation, representing a SPN node.
      void estimateErrorSum(mlir::spn::high::SumNode op);

      /// Estimate the error introduced by the given multiplication operation w.r.t. the current format.
      /// \param op Pointer to the defining operation, representing a SPN node.
      void estimateErrorProduct(mlir::spn::high::ProductNode op);

      /// Estimate the error introduced by the given categrical node w.r.t. the current format.
      /// \param op Pointer to the defining operation, representing a SPN node.
      void estimateErrorCategorical(mlir::spn::high::CategoricalNode op);

      /// Estimate the error introduced by the given gaussian node w.r.t. the current format.
      /// \param op Pointer to the defining operation, representing a SPN node.
      void estimateErrorGaussian(mlir::spn::high::GaussianNode op);

      /// Estimate the error introduced by the given leaf w.r.t. the current format.
      /// \param op Pointer to the defining operation, representing a SPN node.
      void estimateErrorHistogram(mlir::spn::high::HistogramNode op);

      /// Emit diagnostic data / information on the results of this analysis.
      void emitDiagnosticInfo();

      /// Select the "optimal" type, using the current state of the analysis.
      void selectOptimalType();

      /// SPN root-node.
      Operation* root{};

      /// User-requested data representation (e.g. fixed / floating point).
      enum data_representation est_data_representation;

      /// User-requested error model / calculation, also: 'error kind' (e.g. absolute / relative / absolute log).
      enum mlir::spn::high::error_model est_error_model;

      /// User-requested maximum error margin.
      double error_margin{};

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

      /// Calculation constant "two".
      static constexpr double BASE_TWO = 2.0;

      /// Pointer to this analysis' data representation dependant (error) calculator.
      std::shared_ptr<spn::detail::ErrorEstimationCalculator> calc = nullptr;

      /// Calculation of Gaussian minimum probability value outside of 99% := exp(-0.5 * std::pow(2.575829303549,2.0)).
      static constexpr double GAUSS_99 = 0.036245200715160;

      /// For each operation store value's characteristics: accurate, defective, max, min, maximum depth within the SPN.
      std::map<mlir::Operation*, spn::detail::ErrorEstimationValue> spn_node_values;

      /// The global extreme-values of the SPN are used when determining the needed I(nteger) / E(xponent) values
      double spn_node_value_global_maximum = std::numeric_limits<double>::min();
      double spn_node_value_global_minimum = std::numeric_limits<double>::max();

      /// Current format information: "Fraction / Mantissa"
      int format_bits_significance = std::numeric_limits<int>::min();
      /// Current format information: "Integer / Exponent"
      int format_bits_magnitude = std::numeric_limits<int>::min();

      /// Tuples which model different floating point formats, like e.g. fp32 / single precision
      const std::vector<mlir::spn::detail::ValueFormat> Float_Formats{
          spn::detail::ValueFormat{5, 10, Float16Type::get},
          spn::detail::ValueFormat{8, 7, BFloat16Type::get},
          spn::detail::ValueFormat{8, 23, Float32Type::get},
          spn::detail::ValueFormat{11, 52, Float64Type::get}
      };

    };

  }

}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_HISPNTOLOSPN_ARITHMETICPRECISIONANALYSIS_H
