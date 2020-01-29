#include "codegen/shared/Packer.h"

class PackingHeuristic : public Packer {
public:
  vectorizationResultInfo getVectorization(IRGraph &graph, size_t width);
};
