//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_FRONTEND_MLIRDESERIALIZER_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_FRONTEND_MLIRDESERIALIZER_H

#include "mlir/IR/BuiltinOps.h"
#include <util/FileSystem.h>
#include <mlir/IR/Builders.h>
#include <driver/Job.h>
#include "driver/Actions.h"
#include "xspn/xspn/serialization/binary/capnproto/spflow.capnp.h"
#include "llvm/ADT/IndexedMap.h"
#include "HiSPN/HiSPNDialect.h"
#include "HiSPN/HiSPNOps.h"
#include "HiSPN/HiSPNEnums.h"

namespace spnc {

  using bucket_t = std::tuple<int, int, double>;

  class MLIRDeserializer : public ActionWithOutput<::mlir::ModuleOp> {

  public:

    MLIRDeserializer(BinarySPN _inputFile, std::shared_ptr<::mlir::MLIRContext> _context,
                     std::shared_ptr<KernelInfo> info);

    mlir::ModuleOp& execute() override;

  private:

    void deserializeQuery(Query::Reader&& query);

    void deserializeJointQuery(JointProbability::Reader&& query, int batchSize,
                               mlir::spn::high::error_model errorKind, double maxError);

    void deserializeModel(Model::Reader&& model);

    void deserializeNode(Node::Reader& node);

    mlir::spn::high::SumNode deserializeSum(SumNode::Reader&& sum);

    mlir::spn::high::ProductNode deserializeProduct(ProductNode::Reader&& product);

    mlir::spn::high::HistogramNode deserializeHistogram(HistogramLeaf::Reader&& histogram);

    mlir::spn::high::GaussianNode deserializeGaussian(GaussianLeaf::Reader&& gaussian);

    mlir::spn::high::CategoricalNode deserializeCaterogical(CategoricalLeaf::Reader&& categorical);

    mlir::Value getValueForNode(int id);

    mlir::Value getInputValueByIndex(int index);

    mlir::Type translateTypeString(const std::string& text);

    unsigned sizeInByte(mlir::Type type);

    std::shared_ptr<KernelInfo> kernelInfo;

    std::shared_ptr<::mlir::MLIRContext> context;

    std::unique_ptr<::mlir::ModuleOp> module;

    mlir::OpBuilder builder;

    llvm::DenseMap<int, mlir::Value> node2value;

    llvm::DenseMap<int, mlir::Value> inputs;

    BinarySPN inputFile;

    bool cached = false;

  private:

    class BinaryFileHandler {

    public:

      explicit BinaryFileHandler(const std::string& fileName);

      BinaryFileHandler(const BinaryFileHandler&) = delete;

      BinaryFileHandler(BinaryFileHandler&&) = delete;

      BinaryFileHandler& operator=(const BinaryFileHandler&) = delete;

      BinaryFileHandler& operator=(BinaryFileHandler&&) = delete;

      ~BinaryFileHandler() noexcept;

      int& getFileDescriptor() { return fd; }

    private:

      int fd;

    };

  };

}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_FRONTEND_MLIRDESERIALIZER_H
