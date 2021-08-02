//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_ANALYSIS_H
#define SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_ANALYSIS_H

#include "SLPGraph.h"
#include "Util.h"
#include <fstream>

namespace mlir {
  namespace spn {
    namespace low {
      namespace slp {

        void appendLineToFile(StringRef filename, StringRef line) {
          std::ofstream outfile;
          outfile.open(filename.str(), std::ios_base::app);
          assert(outfile.good() && "could not open file");
          outfile << line.str() << "\n";
        }

        void appendCSVLineToFile(StringRef filename, StringRef header, StringRef line, bool alwaysPrintHeader = false) {
          std::ofstream outfile;
          bool exists = std::ifstream{filename.str()}.good();
          outfile.open(filename.str(), std::ios_base::app);
          assert(outfile.good() && "could not open file");
          if (!exists || alwaysPrintHeader) {
            outfile << header.str() << "\n";
          }
          outfile << line.str() << "\n";
        }

        void analyzeTopologicalMixing(SLPGraph const& graph) {
          DenseMap<Value, unsigned> depths;
          llvm::SmallSetVector<Value, 32> worklist;
          for (auto& op : graph.getRoot()->getElement(0).getParentRegion()->getOps()) {
            for (auto const& result : op.getResults()) {
              worklist.insert(result);
              depths[result] = 0;
            }
          }

          while (!worklist.empty()) {
            auto element = worklist.pop_back_val();
            if (auto* definingOp = element.getDefiningOp()) {
              for (auto const& operand : definingOp->getOperands()) {
                if (depths[element] + 1 > depths[operand]) {
                  depths[operand] = depths[element] + 1;
                  worklist.insert(operand);
                }
              }
            }
          }

          DenseMap<Superword*, unsigned> superwordDepths;
          unsigned maxSuperwordDepth = 0;
          llvm::SmallSetVector<Superword*, 32> graphWorklist;
          graphWorklist.insert(graph.getRoot().get());
          superwordDepths[graph.getRoot().get()] = 0;

          while (!graphWorklist.empty()) {
            auto* superword = graphWorklist.pop_back_val();
            for (auto* operand : superword->getOperands()) {
              if (superwordDepths[superword] + 1 > superwordDepths[operand]) {
                superwordDepths[operand] = superwordDepths[superword] + 1;
                graphWorklist.insert(operand);
                maxSuperwordDepth = std::max(superwordDepths[superword] + 1, maxSuperwordDepth);
              }
            }
          }

          auto order = graph::postOrder(graph.getRoot().get());
          unsigned topologicallyMixed = 0;
          unsigned lowest = std::numeric_limits<unsigned>::max();
          for (auto* superword : order) {
            for (auto const& element : *superword) {
              if (depths[superword->getElement(0)] != depths[element]) {
                if (superword->constant() || (superword->uniform() && superword->getElement(0).isa<BlockArgument>())) {
                  continue;
                }
                ++topologicallyMixed;
                lowest = std::min(superwordDepths[superword], lowest);
                break;
              }
            }
          }
          auto header =
              "#ops in function,#ops in graph,width,#superwords,#mixed superwords,lowest mixed occurrence,max depth";
          auto line = Twine(graph.getRoot()->getElement(0).getParentBlock()->getOperations().size())
              .concat(",")
              .concat(std::to_string(numUniqueOps(order)))
              .concat(",")
              .concat(std::to_string(graph.getRoot()->numLanes()))
              .concat(",")
              .concat(std::to_string(order.size()))
              .concat(",")
              .concat(std::to_string(topologicallyMixed))
              .concat(",")
              .concat(lowest == std::numeric_limits<unsigned>::max() ? "none" : std::to_string(lowest))
              .concat(",")
              .concat(std::to_string(maxSuperwordDepth))
              .concat(",")
              .str();

          appendCSVLineToFile("topologicalMixingAnalysis.csv", header, line);

        }

      }
    }
  }
}

#endif //SPNC_MLIR_INCLUDE_CONVERSION_LOSPNTOCPU_VECTORIZATION_SLP_ANALYSIS_H
