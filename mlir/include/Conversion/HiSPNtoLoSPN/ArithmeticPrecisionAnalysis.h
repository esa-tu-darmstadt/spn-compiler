//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_HISPNTOLOSPN_ARITHMETICPRECISIONANALYSIS_H
#define SPNC_MLIR_INCLUDE_CONVERSION_HISPNTOLOSPN_ARITHMETICPRECISIONANALYSIS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "HiSPN/HiSPNOps.h"
#include "LoSPN/LoSPNTypes.h"

namespace mlir {
  namespace spn {

    ///
    /// Analysis performing an arithmetic error estimation
    /// on a graph to determine the best datatype to use
    /// for computation.
    class ArithmeticPrecisionAnalysis {

    public:

      explicit ArithmeticPrecisionAnalysis(Operation* op) : root{op} {
        // TODO Implement to perform the actual analysis on the Graphs
        // contained in the Query.
      }

      mlir::Type getComputationType(bool useLogSpace) {
        // TODO Implement to actually use the analysis results.
        if (useLogSpace) {
          return mlir::spn::low::LogType::get(mlir::FloatType::getF32(root->getContext()));
        }
        return mlir::FloatType::getF64(root->getContext());
      }

    private:

      Operation* root;

    };

  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_HISPNTOLOSPN_ARITHMETICPRECISIONANALYSIS_H
