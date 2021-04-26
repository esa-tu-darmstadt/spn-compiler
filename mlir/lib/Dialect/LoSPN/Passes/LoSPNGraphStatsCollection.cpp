//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <fstream>
#include "LoSPNPassDetails.h"
#include "LoSPN/LoSPNPasses.h"
#include "LoSPN/Analysis/SPNGraphStatistics.h"
#include "LoSPN/Analysis/SPNNodeLevel.h"
#include <util/json.hpp>

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::low;
using json = nlohmann::json;

namespace {

  struct LoSPNGraphStatsCollection : public PassWrapper<LoSPNGraphStatsCollection, OperationPass<ModuleOp>> {
  public:
    LoSPNGraphStatsCollection(std::string graphStatsFilename) : graphStatsFile{std::move(graphStatsFilename)} {}

  protected:
    void runOnOperation() override {
      // This pass does not perform transformations; no analysis will be invalidated.
      markAllAnalysesPreserved();
      // Retrieve the root (ModuleOp)
      auto root = getOperation();
      llvm::SmallVector<Operation*, 5> spn_body_bb;
      root.walk([&spn_body_bb](Operation* op) {
        if (auto spnBody = dyn_cast<SPNBody>(op)) {
          spn_body_bb.push_back(op);
        }
      });

      auto nodeLevel = std::make_unique<SPNNodeLevel>(root);
      auto graphStats = std::make_unique<SPNGraphStatistics>(root);

      // ToDo: Is this a "correct way" of obtaining the featureCount from loSPN?
      auto featureCount = (!spn_body_bb.empty()) ? spn_body_bb.front()->getNumOperands() : -1;

      auto sumCount = graphStats->getKindNodeCount<SPNAdd>();
      auto prodCount = graphStats->getKindNodeCount<SPNMul>();
      auto categCount = graphStats->getKindNodeCount<SPNCategoricalLeaf>();
      auto constCount = graphStats->getKindNodeCount<SPNConstant>();
      auto gaussCount = graphStats->getKindNodeCount<SPNGaussianLeaf>();
      auto histCount = graphStats->getKindNodeCount<SPNHistogramLeaf>();
      auto innerCount = graphStats->getInnerNodeCount();
      auto leafCount = graphStats->getLeafNodeCount();

      auto maxDepth = nodeLevel->getMaxDepth();
      auto minDepth = nodeLevel->getMinDepth();
      auto medianDepth = nodeLevel->getMedianDepth();
      auto avgDepth = nodeLevel->getAverageDepth();

      std::string emitHeader = "\n====================================\n";
      llvm::raw_string_ostream rso{emitHeader};
      rso << "|         LoSPN Statistics         |\n";
      rso << "====================================\n";
      rso << " > Feature count: " << featureCount << "\n";
      rso << " > Minimum depth: " << minDepth << "\n";
      rso << " > Maximum depth: " << maxDepth << "\n";
      rso << " > Average depth: " << avgDepth << "\n";
      rso << " > Median depth:  " << medianDepth << "\n";
      rso << "  -- -- -- -- -- -- -- -- -- -- --\n";
      rso << " > Nodes (inner): " << innerCount << "\n";
      rso << " > Nodes (leaf):  " << leafCount << "\n";
      rso << " > Sum:           " << sumCount << "\n";
      rso << " > Product:       " << prodCount << "\n";
      rso << " > Categorical:   " << categCount << "\n";
      rso << " > Constant:      " << constCount << "\n";
      rso << " > Gaussian:      " << gaussCount << "\n";
      rso << " > Histogram:     " << histCount << "\n";
      rso << "====================================" << "\n";

      root->emitRemark("LoSPN graph statistics collection done.").attachNote(root->getLoc()) << rso.str();

      json stats;

      stats["featureCount"] = featureCount;
      stats["minDepth"] = minDepth;
      stats["maxDepth"] = maxDepth;
      stats["averageDepth"] = avgDepth;
      stats["medianDepth"] = medianDepth;
      stats["innerCount"] = innerCount;
      stats["leafCount"] = leafCount;
      stats["sumCount"] = sumCount;
      stats["productCount"] = prodCount;
      stats["categoricalCount"] = categCount;
      stats["constantCount"] = constCount;
      stats["gaussianCount"] = gaussCount;
      stats["histogramCount"] = histCount;

      std::ofstream fileStream;
      fileStream.open(graphStatsFile);
      fileStream << stats;
      fileStream.close();
    }

  private:
    std::string graphStatsFile;

  };

}

std::unique_ptr<OperationPass<ModuleOp>> mlir::spn::low::createLoSPNGraphStatsCollectionPass(const std::string& graphStatsFile) {
  return std::make_unique<LoSPNGraphStatsCollection>(graphStatsFile);
}
