//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPFUNCTIONTRANSFORMER_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPFUNCTIONTRANSFORMER_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Builders.h"
#include "LoSPNtoCPU/LoSPNtoCPUTypeConverter.h"
#include "LoSPNtoCPU/Vectorization/LoSPNVectorizationTypeConverter.h"
#include "SLPNode.h"
#include <map>
#include <set>
#include <unordered_map>

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        class SLPFunctionTransformer {

        public:

          SLPFunctionTransformer(std::unique_ptr<SLPNode>&& graph, MLIRContext* context);

          void transform();

        private:

          Value transform(SLPNode* node, size_t vectorIndex);
          Value applyCreation(SLPNode* node, size_t vectorIndex, Operation* vectorOp);

          std::unique_ptr<SLPNode> root;
          OpBuilder builder;
          LoSPNVectorizationTypeConverter typeConverter;

          /// Stores where operations can find their operands in case their defining operations
          /// were deleted during vectorization.
          llvm::DenseMap<Operation*, std::map<size_t, std::pair<Operation*, size_t>>> operandExtractions;
          llvm::DenseMap<std::tuple<Value, Value, unsigned>, Value> memRefLoads;

          llvm::DenseMap<SLPNode*, size_t> vectorsDone;
          llvm::DenseMap<SLPNode*, size_t> nodeInputsDone;
          std::set<Operation*> vectorizedOps;

        };
      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPFUNCTIONTRANSFORMER_H
