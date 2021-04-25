//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPCONVERSION_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPCONVERSION_H

#include "SLPNode.h"

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        enum CreationMode {
          Constant,
          Splat,
          BroadcastInsert,
          ConsecutiveLoad,
          Default
        };

        // Storage for vector information (in anonymous namespace to hide it).
        namespace {
          struct NodeVectorData {
            /// The operation that was created for this node vector.
            Optional<Value> operation;
            /// The way it was created.
            Optional<CreationMode> mode;
            /// The first (i.e. smaller Loc) escaping use for each lane.
            DenseMap<size_t, Value> firstEscapingUses;
          };
        }

        class ConversionState {

        public:

          explicit ConversionState(SLPNode* root);

          Value getInsertionPoint(NodeVector* vector) const;
          void update(NodeVector* vector, Value const& operation, CreationMode const& mode);
          bool isConverted(NodeVector* vector) const;

          Value getValue(NodeVector* vector) const;
          CreationMode getCreationMode(NodeVector* vector) const;
          Optional<Value> getFirstEscapingUse(NodeVector* vector, size_t lane) const;

        private:
          DenseMap<NodeVector*, NodeVectorData> vectorData;
          Value earliestInsertionPoint = nullptr;
        };
      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_SLPCONVERSION_H
