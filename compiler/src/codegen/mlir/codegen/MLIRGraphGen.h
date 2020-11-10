//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_CODEGEN_MLIRGRAPHGEN_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_CODEGEN_MLIRGRAPHGEN_H

#include <graph-ir/transform/BaseVisitor.h>
#include <unordered_map>
#include "mlir/IR/Builders.h"

using namespace mlir;

namespace spnc {

  using bucket_t = std::tuple<int, int, double>;

  ///
  /// Translation between SPN nodes in the graph-IR and MLIR operations from the SPN dialect.
  class MLIRGraphGen : public BaseVisitor {

  public:

    MLIRGraphGen(OpBuilder& _builder, std::unordered_map<std::string, Value>& n2v);

    void visitHistogram(Histogram& n, arg_t arg) override;

    void visitProduct(Product& n, arg_t arg) override;

    void visitSum(Sum& n, arg_t arg) override;

    void visitWeightedSum(WeightedSum& n, arg_t arg) override;

  private:

    Value getValueForNode(NodeReference node, arg_t arg);

    OpBuilder& builder;

    std::unordered_map<std::string, Value>& node2value;

  };

}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_CODEGEN_MLIRGRAPHGEN_H
