//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_CODEGENSCALARBODY_H
#define SPNC_CODEGENSCALARBODY_H

#include <unordered_map>
#include <graph-ir/transform/BaseVisitor.h>
#include "CodeGenBody.h"

namespace spnc {

  ///
  /// Code generation for a scalar (i.e., non-vectorized) loop body.
  class CodeGenScalarBody : public CodeGenBody, BaseVisitor {

  public:

    using CodeGenBody::CodeGenBody;

    Value* emitBody(IRGraph& graph, Value* indVar, InputVarValueMap inputs, OutputAddressMap output, const Configuration& config) override;

    void visitHistogram(Histogram& n, arg_t arg) override;

    void visitProduct(Product& n, arg_t arg) override;

    void visitSum(Sum& n, arg_t arg) override;

    void visitWeightedSum(WeightedSum& n, arg_t arg) override;

  private:

    std::unordered_map<std::string, Value*> node2value;

    Type* getValueType();

    void addMetaData(Value* val, TraceMDTag tag);

    Value* getValueForNode(NodeReference node, arg_t arg);

  };
}

#endif //SPNC_CODEGENSCALARBODY_H
