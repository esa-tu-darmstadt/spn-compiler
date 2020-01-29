#include <graph-ir/GraphIRNode.h>
#include "transform/BaseVisitor.h"
#include <unordered_map>
struct SIMDOperationChain {
  std::vector<std::vector<NodeReference>> ops;

};

struct chainNode {
  std::shared_ptr<chainNode> prev;
  NodeReference node;
  unsigned pos;
};

class HeuristicChainBuilder : public BaseVisitor {
 public:
  HeuristicChainBuilder(IRGraph &graph, size_t width);
  std::vector<SIMDOperationChain> getChains();
 private:
  void addNewFront(std::vector<std::shared_ptr<chainNode>>& front, size_t changedLane);
  void buildRoots(std::vector<std::shared_ptr<chainNode>> incompleteFront,
                  size_t width,
                  std::vector<NodeReference>::iterator remainingCandidates,
                  std::vector<NodeReference>::iterator vecEnd);

  void visitHistogram(Histogram &n, arg_t arg) override;

  void visitProduct(Product &n, arg_t arg) override;

  void visitSum(Sum &n, arg_t arg) override;

  void visitWeightedSum(WeightedSum &n, arg_t arg) override;

  template <class T> void visitNode(T &n) {

    for (auto &idx : nodeToFronts[n.id()]) {
      std::vector<std::shared_ptr<chainNode>> front = fronts[idx.first].first;

      int height = front[0]->pos;
      if (height < 2)
        continue;
      bool same = true;
      for (int i = 1; i < front.size(); i++) {
        if (front[i]->pos != height) {
          same = false;
          break;
        }
      }

      if (!same)
        continue;
      chainFronts.push_back(front);
      /*
      SIMDOperationChain chain;
      while (front[0].get() != nullptr) {
        std::vector<NodeReference> op;
        for (int i = 0; i < front.size(); i++) {
          op.push_back(front[i]->node);
          front[i] = front[i]->prev;
        }
        chain.ops.push_back(op);
      }
      chains.push_back(chain);
      */
    }

    // Add new Fronts "starting" at _n_
    for (unsigned i = 2; i <= width; i++) {
      auto rootNodes = getAtLeastNEqualHeightNodes(&n, i);
      buildRoots({}, i, rootNodes.begin(), rootNodes.end());
    }

    for (int i = 0; i < getInputLength(n); i++) {
      auto input = getInput(n, i);
      for (auto &idx : nodeToFronts[n.id()]) {
        std::vector<std::shared_ptr<chainNode>> front = fronts[idx.first].first;
        size_t lastHandledIdx = fronts[idx.first].second;
        // Do not add input to all fronts, just to those where the new front
        // can still be completed
        // -> If the new entry would be higher than an already handled lane,
        // the front won't be ever be completed, thus there is no need to
        // create it in the first place
        if (lastHandledIdx != idx.second &&
            front[idx.second]->pos + 1 > front[lastHandledIdx]->pos)
          continue;
        auto newLane = std::make_shared<chainNode>(
            chainNode{front[idx.second], input, front[idx.second]->pos + 1});
        front[idx.second] = newLane;
        addNewFront(front, idx.second);
      }
      input->accept(*this, {});
    }
  }

  size_t width;

  std::vector<SIMDOperationChain> chains;

  // pair is <lanes of front, last edited front>
  std::vector<std::pair<std::vector<std::shared_ptr<chainNode>>, size_t>> fronts;
  // pair is <front, position in front>
  std::unordered_map<std::string, std::vector<std::pair<size_t, size_t>>> nodeToFronts;

  std::vector<std::vector<std::shared_ptr<chainNode>>> chainFronts;

};
