
#include "codegen/shared/Packer.h"

class PackingTrivial : public Packer {
public:
  vectorizationResultInfo getVectorization(IRGraph &graph, size_t width);

private:
  std::vector<std::vector<NodeReference>>
  getLongestChain(std::vector<NodeReference> vectorRoots,
                  std::unordered_set<std::string> pruned, size_t width);
};
