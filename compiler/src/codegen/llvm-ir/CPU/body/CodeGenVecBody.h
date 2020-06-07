//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_CODEGENVECBODY_H
#define SPNC_CODEGENVECBODY_H

#include <unordered_map>
#include <graph-ir/transform/BaseVisitor.h>
#include "CodeGenBody.h"

namespace spnc {

  ///
  /// Code generation for a scalar (i.e., non-vectorized) loop body.
  class CodeGenVecBody : public CodeGenBody {
  public:
    
    using CodeGenBody::CodeGenBody;
    
    Value* emitBody(IRGraph& graph, Value* indVar, InputVarValueMap inputs, OutputAddressMap output, const Configuration& config) override;
  };
}

#endif
