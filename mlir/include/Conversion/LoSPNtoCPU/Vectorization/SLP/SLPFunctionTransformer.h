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
#include <unordered_set>

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        class SLPFunctionTransformer {

        public:

          SLPFunctionTransformer(std::unique_ptr<SLPNode>&& graph, FuncOp function);

          void transform();

        private:

          Value transform(SLPNode* node, size_t vectorIndex);
          Value applyCreation(SLPNode* node, size_t vectorIndex, Operation* vectorOp, bool keepFirst = false);

          Value extractMemRefOperand(Operation* op);
          Operation* broadcastFirstInsertRest(Operation* beforeOp,
                                              Type const& vectorType,
                                              SmallVector<Value, 4>& elements);
          Value getOrCreateConstant(unsigned index, bool asIndex = true);

          std::unique_ptr<SLPNode> root;
          FuncOp function;

          OpBuilder builder;
          LoSPNVectorizationTypeConverter typeConverter;

          /// Stores where operations can find their operands in case their defining operations
          /// were deleted during vectorization.
          llvm::DenseMap<Operation*, std::map<size_t, std::pair<Operation*, size_t>>> operandExtractions;
          llvm::DenseMap<std::tuple<Value, Value, unsigned>, Value> memRefLoads;
          llvm::DenseMap<unsigned, Value> createdIndexConstants;
          llvm::DenseMap<unsigned, Value> createdUnsignedConstants;

          llvm::DenseMap<SLPNode*, size_t> vectorsDone;
          llvm::DenseMap<SLPNode*, size_t> nodeInputsDone;
          std::unordered_set<Operation*> vectorizedOps;

        };
      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPFUNCTIONTRANSFORMER_H
