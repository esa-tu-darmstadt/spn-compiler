//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <fstream>
#include "LoSPNPassDetails.h"
#include "LoSPN/LoSPNPasses.h"
#include "LoSPN/Analysis/SPNGraphStatistics.h"
#include "LoSPN/Analysis/SPNNodeLevel.h"
// FixMe: Correct include!
#include "../../../../../common/include/util/json.hpp"

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::low;
using json = nlohmann::json;

namespace {

  struct LoSPNGraphStatsCollection : public PassWrapper<LoSPNGraphStatsCollection, OperationPass<ModuleOp>> {
  public:
    LoSPNGraphStatsCollection() { llvm::dbgs() << "Test (1)\n"; llvm::dbgs().flush(); }
    LoSPNGraphStatsCollection(const std::string& graphStatsFilename) : graphStatsFile{graphStatsFilename} {
      llvm::dbgs() << "Test (2)\n"; llvm::dbgs().flush();
    }

  protected:
    void runOnOperation() override {
      llvm::dbgs() << "Test (3)\n"; llvm::dbgs().flush();
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
      auto featureCount = spn_body_bb.front()->getNumOperands();

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
      rso << " > Number of features: " << featureCount << "\n";
      rso << " > Minimum depth: " << minDepth << "\n";
      rso << " > Maximum depth: " << maxDepth << "\n";
      rso << " > Average depth: " << avgDepth << "\n";
      rso << " > Median depth:  " << medianDepth << "\n";
      rso << " > Nodes (inner, leaf): (" << innerCount << ", " << leafCount << ")\n";
      rso << " > Sum-Nodes:         " << sumCount << "\n";
      rso << " > Product-Nodes:     " << prodCount << "\n";
      rso << " > Categorical-Nodes: " << categCount << "\n";
      rso << " > Constant-Nodes:    " << constCount << "\n";
      rso << " > Gaussian-Nodes:    " << gaussCount << "\n";
      rso << " > Histogram-Nodes:   " << histCount << "\n";
      rso << "====================================" << "\n";

      root->emitRemark("Graph statistics collection done.").attachNote(root->getLoc()) << rso.str();

      llvm::dbgs() << "Filename: '" << graphStatsFile << "'\n";

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
  llvm::dbgs() << "Test (0)\n"; llvm::dbgs().flush();
  return std::make_unique<LoSPNGraphStatsCollection>();
}
